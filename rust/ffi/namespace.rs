use std::ffi::{c_char, c_void, CString};
use std::ptr;
use std::sync::Arc;

use arrow::ffi::FFI_ArrowSchema;
use lance::dataset::builder::DatasetBuilder;
use lance_core::Error as LanceError;

use lance_namespace::models::{
    DeclareTableRequest, DescribeTableRequest, DropTableRequest, ListTablesRequest,
};
use lance_namespace::LanceNamespace;
use lance_namespace_impls::RestNamespaceBuilder;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::DatasetHandle;
use super::util::{cstr_to_str, schema_to_ffi_arrow_schema, to_c_string, FfiError, FfiResult};

unsafe fn optional_cstr_to_string(
    ptr: *const c_char,
    what: &'static str,
) -> FfiResult<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = unsafe { cstr_to_str(ptr, what)? };
    if s.is_empty() {
        return Ok(None);
    }
    Ok(Some(s.to_string()))
}

fn parse_headers_tsv(headers_tsv: Option<&str>) -> Vec<(String, String)> {
    headers_tsv
        .map(|tsv| {
            tsv.lines()
                .filter_map(|line| {
                    let mut parts = line.splitn(2, '\t');
                    match (parts.next(), parts.next()) {
                        (Some(k), Some(v)) if !k.is_empty() => Some((k.to_string(), v.to_string())),
                        _ => None,
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn build_config(
    endpoint: &str,
    bearer_token: Option<&str>,
    api_key: Option<&str>,
    headers_tsv: Option<&str>,
) -> RestNamespaceBuilder {
    let mut builder = RestNamespaceBuilder::new(endpoint);
    if let Some(token) = bearer_token {
        builder = builder.header("Authorization", format!("Bearer {token}"));
    }
    if let Some(key) = api_key {
        builder = builder.header("x-api-key", key.to_string());
    }
    // Add custom headers from TSV
    for (key, value) in parse_headers_tsv(headers_tsv) {
        builder = builder.header(key, value);
    }
    builder
}

fn storage_options_to_tsv(storage_options: std::collections::HashMap<String, String>) -> String {
    if storage_options.is_empty() {
        return String::new();
    }
    let mut items: Vec<(String, String)> = storage_options.into_iter().collect();
    items.sort_by(|(a, _), (b, _)| a.cmp(b));
    items
        .into_iter()
        .map(|(k, v)| format!("{k}\t{v}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Max concurrent describe_table requests during prefetch.
const PREFETCH_CONCURRENCY: usize = 10;

/// List tables and prefetch their schemas concurrently.
///
/// Returns `Vec<String>` where each entry is either:
/// - `"table_name\tschema_json"` (schema prefetched successfully), or
/// - `"table_name"` (schema fetch failed or skipped).
///
/// The C++ side parses the tab separator to populate a schema cache.
fn list_tables_inner(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<Vec<String>> {
    let endpoint_s = unsafe { cstr_to_str(endpoint, "endpoint")? }.to_string();
    let namespace_id_s = unsafe { cstr_to_str(namespace_id, "namespace_id")? }.to_string();
    let delimiter_s = unsafe { optional_cstr_to_string(delimiter, "delimiter")? }
        .unwrap_or_else(|| "$".to_string());
    let bearer_token_s = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key_s = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv_s = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let results = runtime::block_on(async {
        // Phase 1: list all table names.
        let list_ns = build_config(
            &endpoint_s,
            bearer_token_s.as_deref(),
            api_key_s.as_deref(),
            headers_tsv_s.as_deref(),
        )
        .delimiter(delimiter_s.clone())
        .build();

        let mut table_names = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let mut req = ListTablesRequest::new();
            req.id = Some(if namespace_id_s.is_empty() {
                Vec::new()
            } else {
                vec![namespace_id_s.clone()]
            });
            req.page_token = page_token.clone();
            req.limit = Some(1000);
            let resp = list_ns.list_tables(req).await.map_err(|err| {
                FfiError::new(
                    ErrorCode::NamespaceListTables,
                    format!("namespace list_tables: {err}"),
                )
            })?;
            table_names.extend(resp.tables);
            match resp.page_token {
                Some(token) if !token.is_empty() => page_token = Some(token),
                _ => break,
            }
        }

        // Phase 2: concurrently describe each table to prefetch schemas.
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(PREFETCH_CONCURRENCY));
        let mut handles = Vec::with_capacity(table_names.len());

        for name in &table_names {
            let ep = endpoint_s.clone();
            let bt = bearer_token_s.clone();
            let ak = api_key_s.clone();
            let delim = delimiter_s.clone();
            let hdr = headers_tsv_s.clone();
            let table_name = name.clone();
            let permit = sem.clone();

            handles.push(tokio::spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                let ns = build_config(&ep, bt.as_deref(), ak.as_deref(), hdr.as_deref())
                    .delimiter(delim)
                    .build();
                let mut req = DescribeTableRequest::new();
                req.id = Some(vec![table_name.clone()]);
                req.load_detailed_metadata = Some(true);
                let schema_json = match ns.describe_table(req).await {
                    Ok(resp) => resp.schema.and_then(|s| serde_json::to_string(&s).ok()),
                    Err(_) => None,
                };
                (table_name, schema_json)
            }));
        }

        let mut out = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok((name, Some(schema))) => out.push(format!("{name}\t{schema}")),
                Ok((name, None)) => out.push(name),
                Err(_) => {} // task panicked, skip
            }
        }
        Ok::<_, FfiError>(out)
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok(results)
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_list_tables(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> *const c_char {
    match list_tables_inner(
        endpoint,
        namespace_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok(tables) => {
            clear_last_error();
            let joined = tables.join("\n");
            to_c_string(joined).into_raw() as *const c_char
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null()
        }
    }
}

fn describe_table_info_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<(String, String)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let (location, storage_options_tsv) = runtime::block_on(async move {
        let mut req = DescribeTableRequest::new();
        req.id = Some(vec![table_id.to_string()]);
        req.with_table_uri = Some(true);
        let resp = namespace.describe_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTableInfo,
                format!("namespace describe_table: {err}"),
            )
        })?;
        let location = resp
            .table_uri
            .or(resp.location)
            .ok_or_else(|| {
                FfiError::new(
                    ErrorCode::NamespaceDescribeTableInfo,
                    "namespace describe_table: missing location and table_uri",
                )
            })?;
        let storage_options_tsv = storage_options_to_tsv(resp.storage_options.unwrap_or_default());
        Ok::<_, FfiError>((location, storage_options_tsv))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok((location, storage_options_tsv))
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_describe_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_location: *mut *const c_char,
    out_storage_options_tsv: *mut *const c_char,
) -> i32 {
    if !out_location.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_location, ptr::null());
        }
    }
    if !out_storage_options_tsv.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_storage_options_tsv, ptr::null());
        }
    }

    match describe_table_info_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok((location, storage_options_tsv)) => {
            clear_last_error();
            if !out_location.is_null() {
                let c = CString::new(location).unwrap_or_else(|_| to_c_string("invalid location"));
                unsafe {
                    std::ptr::write_unaligned(out_location, c.into_raw() as *const c_char);
                }
            }
            if !out_storage_options_tsv.is_null() {
                let c = CString::new(storage_options_tsv)
                    .unwrap_or_else(|_| to_c_string("invalid storage options"));
                unsafe {
                    std::ptr::write_unaligned(
                        out_storage_options_tsv,
                        c.into_raw() as *const c_char,
                    );
                }
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn create_empty_table_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<(String, String)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let (location, storage_options_tsv) = runtime::block_on(async move {
        let mut req = DeclareTableRequest::new();
        req.id = Some(vec![table_id.to_string()]);
        let resp = namespace.declare_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceCreateEmptyTable,
                format!("namespace declare_table: {err}"),
            )
        })?;
        let location = resp.location.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceCreateEmptyTable,
                "namespace declare_table: missing location",
            )
        })?;
        let storage_options_tsv = storage_options_to_tsv(resp.storage_options.unwrap_or_default());
        Ok::<_, FfiError>((location, storage_options_tsv))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok((location, storage_options_tsv))
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_create_empty_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_location: *mut *const c_char,
    out_storage_options_tsv: *mut *const c_char,
) -> i32 {
    if !out_location.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_location, ptr::null());
        }
    }
    if !out_storage_options_tsv.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_storage_options_tsv, ptr::null());
        }
    }

    match create_empty_table_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok((location, storage_options_tsv)) => {
            clear_last_error();
            if !out_location.is_null() {
                let c = CString::new(location).unwrap_or_else(|_| to_c_string("invalid location"));
                unsafe {
                    std::ptr::write_unaligned(out_location, c.into_raw() as *const c_char);
                }
            }
            if !out_storage_options_tsv.is_null() {
                let c = CString::new(storage_options_tsv)
                    .unwrap_or_else(|_| to_c_string("invalid storage options"));
                unsafe {
                    std::ptr::write_unaligned(
                        out_storage_options_tsv,
                        c.into_raw() as *const c_char,
                    );
                }
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn drop_table_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<()> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    runtime::block_on(async move {
        let mut req = DropTableRequest::new();
        req.id = Some(vec![table_id.to_string()]);
        match namespace.drop_table(req).await {
            Ok(_) => Ok(()),
            Err(LanceError::NotFound { .. }) => Ok(()),
            Err(err) => Err(FfiError::new(
                ErrorCode::NamespaceDropTable,
                format!("namespace drop_table '{table_id}': {err}"),
            )),
        }
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_drop_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> i32 {
    match drop_table_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

/// Describe a table with `load_detailed_metadata=true` and return the schema
/// as a JSON string. This avoids opening the dataset from S3.
fn describe_table_with_schema_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<String> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let schema_json = runtime::block_on(async move {
        let mut req = DescribeTableRequest::new();
        req.id = Some(vec![table_id.to_string()]);
        req.with_table_uri = Some(true);
        req.load_detailed_metadata = Some(true);
        let resp = namespace.describe_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                format!("namespace describe_table: {err}"),
            )
        })?;

        let schema = resp.schema.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                "namespace describe_table: missing schema in response",
            )
        })?;

        serde_json::to_string(&schema).map_err(|err| {
            FfiError::new(
                ErrorCode::SchemaExport,
                format!("failed to serialize schema: {err}"),
            )
        })
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok(schema_json)
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_describe_table_with_schema(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_schema_json: *mut *const c_char,
) -> i32 {
    if !out_schema_json.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_schema_json, ptr::null());
        }
    }

    match describe_table_with_schema_inner(
        endpoint, table_id, bearer_token, api_key, delimiter, headers_tsv,
    ) {
        Ok(schema_json) => {
            clear_last_error();
            if !out_schema_json.is_null() {
                let c = CString::new(schema_json)
                    .unwrap_or_else(|_| to_c_string("invalid schema json"));
                unsafe {
                    std::ptr::write_unaligned(out_schema_json, c.into_raw() as *const c_char);
                }
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn open_dataset_in_namespace_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<(DatasetHandle, String)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let (dataset, table_uri) = runtime::block_on(async move {
        let mut req = DescribeTableRequest::new();
        req.id = Some(vec![table_id.to_string()]);
        req.with_table_uri = Some(true);
        let resp = namespace.describe_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                format!("namespace describe_table: {err}"),
            )
        })?;

        let table_uri = resp
            .table_uri
            .or(resp.location)
            .ok_or_else(|| {
                FfiError::new(
                    ErrorCode::NamespaceDescribeTable,
                    "namespace describe_table: missing location and table_uri",
                )
            })?;
        let storage_options = resp.storage_options.unwrap_or_default();

        let dataset = DatasetBuilder::from_uri(&table_uri)
            .with_storage_options(storage_options)
            .load()
            .await
            .map_err(|err| {
                FfiError::new(
                    ErrorCode::DatasetOpen,
                    format!("dataset open '{table_uri}': {err}"),
                )
            })?;
        Ok::<_, FfiError>((Arc::new(dataset), table_uri))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok((DatasetHandle::new(dataset), table_uri))
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_in_namespace(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_table_uri: *mut *const c_char,
) -> *mut c_void {
    if !out_table_uri.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_table_uri, ptr::null());
        }
    }

    match open_dataset_in_namespace_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok((handle, table_uri)) => {
            clear_last_error();
            if !out_table_uri.is_null() {
                let uri_c = CString::new(table_uri).unwrap_or_else(|_| to_c_string("invalid uri"));
                unsafe {
                    std::ptr::write_unaligned(out_table_uri, uri_c.into_raw() as *const c_char);
                }
            }
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

// ── TMP: local convert_json_arrow_schema that supports nested types ──────────
// Remove once lance-namespace upstream merges the fix:
// https://github.com/lancedb/lance/pull/XXXX
mod tmp_schema {
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
    use lance_core::{Error, Result};
    use lance_namespace::models::{JsonArrowDataType, JsonArrowField, JsonArrowSchema};
    use std::sync::Arc;

    pub fn convert_schema(json_schema: &JsonArrowSchema) -> Result<ArrowSchema> {
        let fields: Result<Vec<Field>> = json_schema.fields.iter().map(convert_field).collect();
        let metadata = json_schema.metadata.as_ref().cloned().unwrap_or_default();
        Ok(ArrowSchema::new_with_metadata(fields?, metadata))
    }

    fn convert_field(f: &JsonArrowField) -> Result<Field> {
        let dt = convert_type(&f.r#type)?;
        let field = Field::new(&f.name, dt, f.nullable);
        Ok(match f.metadata.as_ref() {
            Some(m) => field.with_metadata(m.clone()),
            None => field,
        })
    }

    fn convert_type(t: &JsonArrowDataType) -> Result<DataType> {
        let name = t.r#type.to_lowercase();
        match name.as_str() {
            "null" => Ok(DataType::Null),
            "bool" | "boolean" => Ok(DataType::Boolean),
            "int8" => Ok(DataType::Int8),
            "uint8" => Ok(DataType::UInt8),
            "int16" => Ok(DataType::Int16),
            "uint16" => Ok(DataType::UInt16),
            "int32" => Ok(DataType::Int32),
            "uint32" => Ok(DataType::UInt32),
            "int64" => Ok(DataType::Int64),
            "uint64" => Ok(DataType::UInt64),
            "float16" => Ok(DataType::Float16),
            "float32" => Ok(DataType::Float32),
            "float64" => Ok(DataType::Float64),
            "decimal128" => {
                let e = t.length.unwrap_or(0);
                Ok(DataType::Decimal128((e / 1000) as u8, (e % 1000) as i8))
            }
            "date32" => Ok(DataType::Date32),
            "date64" => Ok(DataType::Date64),
            "timestamp" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
            "duration" => Ok(DataType::Duration(TimeUnit::Microsecond)),
            "utf8" => Ok(DataType::Utf8),
            "large_utf8" => Ok(DataType::LargeUtf8),
            "binary" => Ok(DataType::Binary),
            "large_binary" => Ok(DataType::LargeBinary),
            "fixed_size_binary" => Ok(DataType::FixedSizeBinary(t.length.unwrap_or(0) as i32)),
            "list" => {
                let inner = t.fields.as_ref().and_then(|f| f.first())
                    .ok_or_else(|| Error::namespace("list type missing inner field"))?;
                Ok(DataType::List(Arc::new(convert_field(inner)?)))
            }
            "large_list" => {
                let inner = t.fields.as_ref().and_then(|f| f.first())
                    .ok_or_else(|| Error::namespace("large_list type missing inner field"))?;
                Ok(DataType::LargeList(Arc::new(convert_field(inner)?)))
            }
            "fixed_size_list" => {
                let inner = t.fields.as_ref().and_then(|f| f.first())
                    .ok_or_else(|| Error::namespace("fixed_size_list type missing inner field"))?;
                Ok(DataType::FixedSizeList(Arc::new(convert_field(inner)?), t.length.unwrap_or(0) as i32))
            }
            "struct" => {
                let fields = t.fields.as_ref()
                    .ok_or_else(|| Error::namespace("struct type missing fields"))?;
                let fs: Result<Vec<Field>> = fields.iter().map(convert_field).collect();
                Ok(DataType::Struct(fs?.into()))
            }
            "map" => {
                let entries = t.fields.as_ref().and_then(|f| f.first())
                    .ok_or_else(|| Error::namespace("map type missing entries field"))?;
                Ok(DataType::Map(Arc::new(convert_field(entries)?), false))
            }
            _ => Err(Error::namespace(format!("Unsupported Arrow type: {name}"))),
        }
    }
}
// ── end TMP ─────────────────────────────────────────────────────────────────

/// Convert a JSON Arrow schema string to Arrow C Data Interface ArrowSchema.
#[no_mangle]
pub unsafe extern "C" fn lance_json_arrow_schema_to_c(
    json_schema: *const c_char,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    let result = (|| -> FfiResult<()> {
        let json_str = unsafe { cstr_to_str(json_schema, "json_schema")? };
        let json_arrow: lance_namespace::models::JsonArrowSchema =
            serde_json::from_str(json_str).map_err(|err| {
                FfiError::new(
                    ErrorCode::SchemaExport,
                    format!("failed to parse JSON arrow schema: {err}"),
                )
            })?;
        // TMP: use local schema converter that supports nested types.
        // Switch back to convert_json_arrow_schema once upstream lance-namespace is fixed.
        let arrow_schema = tmp_schema::convert_schema(&json_arrow).map_err(|err| {
            FfiError::new(
                ErrorCode::SchemaExport,
                format!("failed to convert JSON arrow schema: {err}"),
            )
        })?;
        let ffi_schema = schema_to_ffi_arrow_schema(&arrow_schema)?;
        unsafe {
            std::ptr::write_unaligned(out_schema, ffi_schema);
        }
        Ok(())
    })();

    match result {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}
