#!/bin/bash

# Heroku Deployment Script for ChessLytics
echo "🚀 Starting Heroku deployment for ChessLytics..."

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI is not installed. Please install it first:"
    echo "   macOS: brew tap heroku/brew && brew install heroku"
    echo "   Windows: Download from https://devcenter.heroku.com/articles/heroku-cli"
    echo "   Linux: curl https://cli-assets.heroku.com/install.sh | sh"
    exit 1
fi

# Check if user is logged in
if ! heroku auth:whoami &> /dev/null; then
    echo "🔐 Please log in to Heroku first:"
    heroku login
fi

# Get existing app name from user
echo "📝 Enter your existing Heroku app name:"
read app_name

if [ -z "$app_name" ]; then
    echo "❌ App name is required. Please provide your existing Heroku app name."
    exit 1
fi

# Verify the app exists
if ! heroku apps:info --app $app_name &> /dev/null; then
    echo "❌ App '$app_name' not found or you don't have access to it."
    echo "   Please check your app name or create a new app first."
    exit 1
fi

echo "✅ Found existing app: $app_name"

# Set up environment variables
echo "🔧 Setting up environment variables..."

# Check if service account file exists
if [ -f "gcp/service_account.json" ]; then
    echo "📄 Setting Google Cloud credentials..."
    heroku config:set GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat gcp/service_account.json)" --app $app_name
else
    echo "⚠️  Warning: gcp/service_account.json not found. You'll need to set GOOGLE_APPLICATION_CREDENTIALS_JSON manually."
fi

# Set other environment variables
heroku config:set FLASK_ENV=production --app $app_name
heroku config:set FLASK_DEBUG=0 --app $app_name

echo "✅ Environment variables configured"

# Deploy the application
echo "📦 Deploying to Heroku..."
git add .
git commit -m "Deploy to Heroku - $(date)"

echo "🚀 Pushing to Heroku..."
git push heroku main

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Opening your app..."
    heroku open --app $app_name
    
    echo "📊 View logs: heroku logs --tail --app $app_name"
    echo "🔧 Manage app: https://dashboard.heroku.com/apps/$app_name"
else
    echo "❌ Deployment failed. Check the logs above for errors."
    echo "📋 View logs: heroku logs --tail --app $app_name"
fi 