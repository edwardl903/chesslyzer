"""
BigQuery + Looker Studio Integration for Personalized Chess Dashboards
"""
import os
import json
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError, NotFound
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import hashlib
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryDashboardManager:
    def __init__(self):
        """Initialize BigQuery and Looker Studio integration"""
        self.project_id = "crucial-decoder-462021-m4"  # Match the project where data is uploaded
        self.dataset_id = "test1"  # Match the dataset where data is uploaded
        self.games_table = "megachessdataset"  # Match the exact table where data is uploaded
        self.stats_table = "user_statistics"
        
        # Initialize BigQuery client
        self.client = self._get_bigquery_client()
        
        # Looker Studio dashboard template URL
        # Real Looker Studio dashboard - convert edit URL to embed URL
        # Original: https://lookerstudio.google.com/reporting/dbe35905-fe7a-4971-a502-0e0e5fbe7a3d/page/p_44hm6tf7sd/edit
        # Embed: https://lookerstudio.google.com/embed/reporting/dbe35905-fe7a-4971-a502-0e0e5fbe7a3d
        self.dashboard_template_url = "https://lookerstudio.google.com/embed/reporting/dbe35905-fe7a-4971-a502-0e0e5fbe7a3d"
        logger.info("BigQuery Dashboard Manager initialized")
    
    def _get_bigquery_client(self):
        """Get BigQuery client with service account"""
        try:
            # Check for Heroku environment variable first
            if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
                # Parse the JSON from environment variable
                credentials_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                return bigquery.Client(credentials=credentials, project=credentials_info.get('project_id'))
            
            # Fallback to local service account file
            service_account_path = os.path.join(os.path.dirname(__file__), '..', 'gcp', 'service_account.json')
            
            if os.path.exists(service_account_path):
                return bigquery.Client.from_service_account_json(service_account_path)
            else:
                logger.warning("Service account not found. Using default credentials.")
                return bigquery.Client()
                
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    def _ensure_dataset_exists(self):
        """Ensure the dataset exists"""
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {self.dataset_id} exists")
        except NotFound:
            dataset = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
            dataset.location = "US"  # Set your preferred location
            dataset = self.client.create_dataset(dataset)
            logger.info(f"Created dataset: {self.dataset_id}")
    
    def _ensure_tables_exist(self):
        """Ensure BigQuery tables exist with proper schema"""
        self._ensure_dataset_exists()
        
        # Games table schema
        games_schema = [
            bigquery.SchemaField("username", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("time_control", "STRING"),
            bigquery.SchemaField("result", "STRING"),
            bigquery.SchemaField("rating", "INTEGER"),
            bigquery.SchemaField("opponent_rating", "INTEGER"),
            bigquery.SchemaField("opening", "STRING"),
            bigquery.SchemaField("moves", "STRING"),
            bigquery.SchemaField("game_duration", "INTEGER"),  # in seconds
            bigquery.SchemaField("color", "STRING"),
            bigquery.SchemaField("platform", "STRING", default_value_expression="'lichess'"),
            bigquery.SchemaField("created_at", "TIMESTAMP", default_value_expression="CURRENT_TIMESTAMP()")
        ]
        
        # Statistics table schema
        stats_schema = [
            bigquery.SchemaField("username", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("total_games", "INTEGER"),
            bigquery.SchemaField("wins", "INTEGER"),
            bigquery.SchemaField("losses", "INTEGER"),
            bigquery.SchemaField("draws", "INTEGER"),
            bigquery.SchemaField("current_rating", "INTEGER"),
            bigquery.SchemaField("highest_rating", "INTEGER"),
            bigquery.SchemaField("lowest_rating", "INTEGER"),
            bigquery.SchemaField("favorite_opening", "STRING"),
            bigquery.SchemaField("total_time_spent", "INTEGER"),  # in minutes
            bigquery.SchemaField("biggest_win", "STRING"),
            bigquery.SchemaField("biggest_loss", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP", default_value_expression="CURRENT_TIMESTAMP()")
        ]
        
        # Create tables if they don't exist
        for table_name, schema in [(self.games_table, games_schema), (self.stats_table, stats_schema)]:
            table_ref = self.client.dataset(self.dataset_id).table(table_name)
            try:
                self.client.get_table(table_ref)
                logger.info(f"Table {table_name} exists")
            except NotFound:
                table = bigquery.Table(table_ref, schema=schema)
                self.client.create_table(table)
                logger.info(f"Created table: {table_name}")
    
    def upload_user_games(self, username: str, games_data: List[Dict[str, Any]]):
        """Upload user's games to BigQuery"""
        try:
            self._ensure_tables_exist()
            
            if not games_data:
                logger.warning(f"No games data for user {username}")
                return
            
            # Convert games data to DataFrame
            df = pd.DataFrame(games_data)
            
            # Add username and created_at columns
            df['username'] = username
            df['created_at'] = datetime.now()
            
            # Upload to BigQuery - don't specify schema, let BigQuery use table's schema
            table_ref = self.client.dataset(self.dataset_id).table(self.games_table)
            
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND"
                # Remove schema specification to use table's existing schema
            )
            
            load_job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            load_job.result()
            
            logger.info(f"Successfully uploaded {len(games_data)} games for user {username}")
            
        except Exception as e:
            logger.error(f"Failed to upload games for user {username}: {e}")
            raise
    
    def upload_user_statistics(self, username: str, stats_data: Dict[str, Any]):
        """Upload user's statistics to BigQuery"""
        try:
            self._ensure_tables_exist()
            
            # Convert stats to DataFrame
            stats_row = {
                'username': username,
                'date': datetime.now().date(),
                'total_games': stats_data.get('total_games', 0),
                'wins': stats_data.get('wins', 0),
                'losses': stats_data.get('losses', 0),
                'draws': stats_data.get('draws', 0),
                'current_rating': stats_data.get('current_rating', 0),
                'highest_rating': stats_data.get('highest_rating', 0),
                'lowest_rating': stats_data.get('lowest_rating', 0),
                'favorite_opening': stats_data.get('favorite_opening', ''),
                'total_time_spent': stats_data.get('total_time_spent', {}).get('total_minutes', 0),
                'biggest_win': stats_data.get('biggest_win', ''),
                'biggest_loss': stats_data.get('biggest_loss', ''),
                'created_at': datetime.now()
            }
            
            df = pd.DataFrame([stats_row])
            
            # Upload to BigQuery
            table_ref = self.client.dataset(self.dataset_id).table(self.stats_table)
            
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            load_job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            load_job.result()
            
            logger.info(f"Successfully uploaded statistics for user {username}")
            
        except Exception as e:
            logger.error(f"Failed to upload statistics for user {username}: {e}")
            raise
    
    def generate_personalized_dashboard_url(self, username: str, year: str = None) -> str:
        """Generate a personalized Looker Studio dashboard URL for the user"""
        try:
            # Build username_year parameter
            username_year = username
            if year:
                username_year = f"{username}_{year}"
            
            # Use the same dynamic view approach
            view_name = "chess_games_dynamic_view"
            view_id = f"{self.project_id}.{self.dataset_id}.{view_name}"
            
            # Create a URL with parameters that Looker Studio can use for filtering
            dashboard_url = (
                f"{self.dashboard_template_url}"
                f"?ds={self.project_id}.{self.dataset_id}.{view_name}"
                f"&user_filter={username_year}"
            )
            
            logger.info(f"Generated personalized dashboard URL for user {username} with year {year} -> {username_year}")
            return dashboard_url
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard URL for user {username}: {e}")
            # Return a fallback URL
            return f"{self.dashboard_template_url}?user_filter={username}"
    
    def _generate_user_token(self, username: str) -> str:
        """Generate a secure token for dashboard access"""
        # Create a hash based on username and current date
        data = f"{username}_{datetime.now().strftime('%Y-%m-%d')}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_user_data_from_bigquery(self, username: str) -> Dict[str, Any]:
        """Retrieve user data from BigQuery for dashboard generation"""
        try:
            # Query user's games
            games_query = f"""
            SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.games_table}`
            WHERE username = '{username}'
            ORDER BY date DESC
            LIMIT 1000
            """
            
            games_df = self.client.query(games_query).to_dataframe()
            
            # Query user's latest statistics
            stats_query = f"""
            SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.stats_table}`
            WHERE username = '{username}'
            ORDER BY date DESC
            LIMIT 1
            """
            
            stats_df = self.client.query(stats_query).to_dataframe()
            
            return {
                'games': games_df.to_dict('records') if not games_df.empty else [],
                'statistics': stats_df.to_dict('records')[0] if not stats_df.empty else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve data for user {username}: {e}")
            return {'games': [], 'statistics': {}}
    
    def create_user_specific_dashboard_url(self, username: str, year: str = None) -> str:
        """Create a user-specific dashboard URL using a single dynamic view"""
        try:
            username_year = f"{username}_{year}" if year else username
            print(f"🔍 Creating dashboard for user: {username}, year: {year}, username_year: {username_year}")
            
            # Create or update the single dynamic view in BigQuery
            view_name = "chess_games_dynamic_view"
            view_id = f"{self.project_id}.{self.dataset_id}.{view_name}"
            
            # Create a view that includes all data but can be filtered by URL parameters
            # This view will be used by Looker Studio with URL parameters for filtering
            view_query = f"""
            CREATE OR REPLACE VIEW `{view_id}`
            AS SELECT 
                url,
                uuid,
                timestamp,
                time_class,
                game_type,
                rated,
                eco,
                my_opening,
                my_username,
                my_rating,
                my_result,
                my_color,
                opp_username,
                opp_rating,
                opp_result,
                my_win_or_lose,
                rating_diff,
                my_time_left,
                opp_time_left,
                my_time_left_ratio,
                opp_time_left_ratio,
                time_spent,
                my_moves,
                opp_moves,
                my_num_moves,
                en_passant_count,
                promotion_count,
                my_castling,
                opp_castling,
                month,
                weekday,
                hour,
                day_of_week,
                unique_id,
                uploaded_at,
                CONCAT(my_username, '_', EXTRACT(YEAR FROM timestamp)) as username_year
            FROM `{self.project_id}.{self.dataset_id}.{self.games_table}`
            """
            
            # Execute the view creation
            try:
                self.client.query(view_query).result()
                print(f"✅ Updated dynamic view: {view_name}")
                
                # Verify the view has data for this user
                verify_query = f"""
                SELECT COUNT(*) as count 
                FROM `{view_id}` 
                WHERE username_year = '{username_year}'
                """
                result = self.client.query(verify_query).result()
                for row in result:
                    count = row.count
                    print(f"📊 Found {count} records for {username_year} in dynamic view")
                    
            except Exception as e:
                print(f"⚠️ View update warning: {e}")
            
            # Create a URL with parameters that Looker Studio can use for filtering
            # The dashboard will use URL parameters to filter the data
            dashboard_url = (
                f"{self.dashboard_template_url}"
                f"?ds={self.project_id}.{self.dataset_id}.{view_name}"
                f"&user_filter={username_year}"
            )
            
            logger.info(f"Created user-specific dashboard URL for {username_year}: {dashboard_url}")
            print(f"🔗 Generated dashboard URL: {dashboard_url}")
            return dashboard_url
            
        except Exception as e:
            logger.error(f"Failed to create user-specific dashboard: {e}")
            print(f"❌ Error creating dashboard: {e}")
            # Fallback to regular dashboard
            return self.dashboard_template_url
    
    def create_embed_dashboard_url(self, username: str, year: str = None) -> dict:
        """Create a Looker Embed SDK configuration for the user"""
        try:
            username_year = f"{username}_{year}" if year else username
            
            # Create or update the single dynamic view in BigQuery
            view_name = "chess_games_dynamic_view"
            view_id = f"{self.project_id}.{self.dataset_id}.{view_name}"
            
            # Create a view that includes all data but can be filtered by URL parameters
            view_query = f"""
            CREATE OR REPLACE VIEW `{view_id}`
            AS SELECT 
                url,
                uuid,
                date,
                timestamp,
                time_control,
                time_class,
                game_type,
                rated,
                eco,
                my_opening,
                my_username,
                my_rating,
                my_result,
                my_color,
                opp_username,
                opp_rating,
                opp_result,
                my_win_or_lose,
                rating_diff,
                my_time_left,
                opp_time_left,
                my_time_left_ratio,
                opp_time_left_ratio,
                time_spent,
                my_moves,
                opp_moves,
                moves,
                my_num_moves,
                en_passant_count,
                promotion_count,
                my_castling,
                opp_castling,
                month,
                weekday,
                hour,
                day_of_week,
                unique_id,
                uploaded_at,
                CONCAT(my_username, '_', EXTRACT(YEAR FROM timestamp)) as username_year
            FROM `{self.project_id}.{self.dataset_id}.{self.games_table}`
            """
            
            # Execute the view creation
            try:
                self.client.query(view_query).result()
                print(f"✅ Updated dynamic view: {view_name}")
            except Exception as e:
                print(f"⚠️ View update warning: {e}")
            
            # Create embed configuration
            embed_config = {
                "dashboard_id": "dbe35905-fe7a-4971-a502-0e0e5fbe7a3d",  # Your dashboard ID
                "data_source": f"{self.project_id}.{self.dataset_id}.{view_name}",
                "filters": {
                    "username_year": username_year
                },
                "tile_id": None,  # Show full dashboard
                "embed_domain": "chesslytics.xyz",  # Your domain
                "force_logout_login": False,
                "session_length": 3600,  # 1 hour session
                "external_user_id": username,
                "user_attributes": {
                    "username": username,
                    "year": year or "all"
                }
            }
            
            logger.info(f"Created embed configuration for {username_year}")
            return embed_config
            
        except Exception as e:
            logger.error(f"Failed to create embed configuration: {e}")
            print(f"❌ Error creating embed config: {e}")
            return None

    def create_user_specific_view(self, username: str, year: str = None) -> str:
        """Create a user-specific view that only contains that user's data"""
        try:
            username_year = f"{username}_{year}" if year else username
            print(f"🔍 Creating user-specific view for: {username}, year: {year}, username_year: {username_year}")
            
            # Create a user-specific view name
            view_name = f"user_data_{username}_{year}".replace('-', '_').replace('.', '_')
            view_id = f"{self.project_id}.{self.dataset_id}.{view_name}"
            
            # Create a view that only contains this user's data
            view_query = f"""
            CREATE OR REPLACE VIEW `{view_id}`
            AS SELECT 
                *,
                CONCAT(my_username, '_', EXTRACT(YEAR FROM timestamp)) as username_year
            FROM `{self.project_id}.{self.dataset_id}.{self.games_table}`
            WHERE username = '{username}'
            """
            
            if year:
                view_query += f" AND EXTRACT(YEAR FROM date) = {year}"
            
            # Execute the view creation
            try:
                self.client.query(view_query).result()
                print(f"✅ Created user-specific view: {view_name}")
                
                # Verify the view has data
                verify_query = f"""
                SELECT COUNT(*) as count 
                FROM `{view_id}`
                """
                result = self.client.query(verify_query).result()
                for row in result:
                    count = row.count
                    print(f"📊 View contains {count} records for {username_year}")
                    
            except Exception as e:
                print(f"⚠️ View creation warning: {e}")
            
            # Return dashboard URL with the user-specific view
            dashboard_url = (
                f"{self.dashboard_template_url}"
                f"?ds={self.project_id}.{self.dataset_id}.{view_name}"
            )
            
            logger.info(f"Created user-specific view URL for {username_year}: {dashboard_url}")
            print(f"🔗 Generated view URL: {dashboard_url}")
            return dashboard_url
            
        except Exception as e:
            logger.error(f"Failed to create user-specific view: {e}")
            print(f"❌ Error creating user-specific view: {e}")
            # Fallback to regular dashboard
# Global instance
bigquery_dashboard = BigQueryDashboardManager() 