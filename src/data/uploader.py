# bigquery_uploader.py
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError, NotFound
import pandas as pd
import os
import json
from google.oauth2 import service_account
import time


def get_bigquery_client(credentials_path="gcp/service_account.json"):
    if os.environ.get("GOOGLE_CREDENTIALS"):
        # Running in Heroku or cloud: use env var
        credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    else:
        # Running locally: use service_account.json file
        return bigquery.Client.from_service_account_json(credentials_path)

def upload_to_bigquery(
        df, 
        table_id, 
        credentials_path="gcp/service_account.json",
        write_disposition="WRITE_APPEND",
        schema=None
):
    """
    Uploads a DataFrame to BigQuery. Creates dataset/table if needed.

    Args:
        df (pd.DataFrame): Data to upload.
        table_id (str): Full table ID in format 'project.dataset.table'.
        credentials_path (str): Path to service account JSON key.
        write_disposition (str): "WRITE_APPEND", "WRITE_TRUNCATE", or "WRITE_EMPTY".
        schema (list): Optional BigQuery schema as list of SchemaField objects.
    """
    if df.empty:
        print("⚠️ DataFrame is empty. Upload skipped.")
        return

    try:
        client = get_bigquery_client()
        project_id, dataset_id, table_name = table_id.split('.')
        dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
        table_ref = client.dataset(dataset_id).table(table_name)

        # Ensure dataset exists
        try:
            client.get_dataset(dataset_ref)
        except NotFound:
            client.create_dataset(dataset_ref)
            print(f"✅ Created dataset: {dataset_id}")

        # Ensure table exists
        try:
            client.get_table(table_ref)
            print(f"📦 Table already exists: {table_id}")
        except NotFound:
            if schema:
                table = bigquery.Table(table_ref, schema=schema)
                client.create_table(table)
                print(f"✅ Created table with schema: {table_id}")
            else:
                # If no schema, let BigQuery infer it
                job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
                job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
                job.result()
                print(f"✅ Table created and data uploaded: {table_id}")
                return

        # Upload data to existing table
        job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
        if schema:
            job_config.schema = schema

        load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        load_job.result()
        print(f"✅ Successfully uploaded to BigQuery table: {table_id}")

    except GoogleAPIError as e:
        print(f"❌ Google API Error: {e}")
    except Exception as e:
        print(f"❌ Failed to upload to BigQuery: {e}")

def merge_to_bigquery_table(table_id, df, unique_columns=['uuid'], credentials_path="gcp/service_account.json"):
    """
    Uses BigQuery MERGE to efficiently upsert data without duplicates.
    
    Args:
        table_id (str): Full table ID in format 'project.dataset.table'.
        df (pd.DataFrame): Data to merge.
        unique_columns (list): Columns to use for matching records.
        credentials_path (str): Path to service account JSON key.
    """
    try:
        client = get_bigquery_client(credentials_path)
        project_id, dataset_id, table_name = table_id.split('.')
        
        print(f"🔍 Processing table: {table_id}")
        
        # Ensure dataset exists
        dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
        try:
            client.get_dataset(dataset_ref)
            print(f"✅ Dataset exists: {dataset_id}")
        except NotFound:
            client.create_dataset(dataset_ref)
            print(f"✅ Created dataset: {dataset_id}")
        
        # Check if target table exists
        table_ref = client.dataset(dataset_id).table(table_name)
        table_exists = False
        table_has_schema = False
        
        try:
            table = client.get_table(table_ref)
            table_exists = True
            table_has_schema = len(table.schema) > 0
            print(f"📦 Target table exists with {len(table.schema)} fields: {table_id}")
        except NotFound:
            print(f"📦 Target table does not exist: {table_id}")
        
        # Create temporary table for new data
        temp_table_name = f"{table_name}_temp_{int(time.time())}"
        temp_table_ref = client.dataset(dataset_id).table(temp_table_name)
        
        # Upload new data to temp table
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        load_job = client.load_table_from_dataframe(df, temp_table_ref, job_config=job_config)
        load_job.result()
        print(f"✅ Uploaded data to temp table: {temp_table_name}")
        
        if not table_exists or not table_has_schema:
            # If target table doesn't exist OR has no schema, recreate it
            if table_exists:
                print(f"🔄 Table exists but has no schema, recreating: {table_id}")
                client.delete_table(table_ref)
            
            # Copy temp table to create target table
            copy_job = client.copy_table(temp_table_ref, table_ref)
            copy_job.result()
            print(f"✅ Created target table from temp data: {table_id}")
        else:
            # Table exists and has schema, use MERGE for efficient upsert
            print(f"🔄 Using MERGE for existing table: {table_id}")
            
            # Build MERGE query
            unique_cols_str = ', '.join(unique_columns)
            merge_conditions = ' AND '.join([f"T.{col} = S.{col}" for col in unique_columns])
            
            merge_query = f"""
            MERGE `{table_id}` T
            USING `{project_id}.{dataset_id}.{temp_table_name}` S
            ON {merge_conditions}
            WHEN MATCHED THEN
                UPDATE SET
                    uploaded_at = S.uploaded_at
            WHEN NOT MATCHED THEN
                INSERT ROW
            """
            
            print(f"🔍 Executing MERGE query...")
            query_job = client.query(merge_query)
            query_job.result()
            print(f"✅ Successfully merged data to table: {table_id}")
        
        # Clean up temp table
        client.delete_table(temp_table_ref)
        print(f"🧹 Cleaned up temp table: {temp_table_name}")
        
    except Exception as e:
        print(f"❌ Failed to merge data: {e}")
        # Clean up temp table on error
        try:
            client.delete_table(temp_table_ref)
            print(f"🧹 Cleaned up temp table on error: {temp_table_name}")
        except:
            pass
        raise  # Re-raise the exception to see the full error