# Heroku Deployment Guide for ChessLytics

This guide will walk you through deploying your ChessLytics application to Heroku.

## Prerequisites

1. **Heroku Account**: Sign up at [heroku.com](https://heroku.com)
2. **Heroku CLI**: Install from [devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)
3. **Git**: Make sure your project is in a Git repository

## Step 1: Install Heroku CLI

### macOS (using Homebrew):
```bash
brew tap heroku/brew && brew install heroku
```

### Windows:
Download and run the installer from the Heroku website.

### Linux:
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

## Step 2: Login to Heroku

```bash
heroku login
```

This will open your browser to authenticate with Heroku.

## Step 3: Prepare Your Application

Your application is already prepared with the necessary files:
- ✅ `Procfile` - Tells Heroku how to run your app
- ✅ `requirements.txt` - Lists all Python dependencies
- ✅ `app.py` - Your Flask application

## Step 4: Create a Heroku App

Navigate to your project directory and create a new Heroku app:

```bash
cd chesslyzer_wip
heroku create your-chesslytics-app-name
```

Replace `your-chesslytics-app-name` with a unique name for your app.

## Step 5: Set Up Environment Variables

You'll need to configure your Google Cloud credentials as environment variables:

```bash
# Set your Google Cloud service account key
heroku config:set GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat gcp/service_account.json)"

# Set any other environment variables you need
heroku config:set FLASK_ENV=production
```

## Step 6: Deploy Your Application

```bash
# Add all files to git (if not already done)
git add .

# Commit your changes
git commit -m "Deploy to Heroku"

# Push to Heroku
git push heroku main
```

If your default branch is `master` instead of `main`, use:
```bash
git push heroku master
```

## Step 7: Open Your Application

```bash
heroku open
```

This will open your deployed application in your browser.

## Step 8: Monitor Your Application

```bash
# View logs
heroku logs --tail

# Check app status
heroku ps

# Scale your app (if needed)
heroku ps:scale web=1
```

## Troubleshooting

### Common Issues:

1. **Build Fails**: Check the logs with `heroku logs --tail`
2. **App Crashes**: Make sure all dependencies are in `requirements.txt`
3. **Environment Variables**: Verify they're set correctly with `heroku config`

### Useful Commands:

```bash
# View all config variables
heroku config

# Run the app locally with Heroku environment
heroku local web

# Access Heroku bash
heroku run bash

# Restart the app
heroku restart
```

## Updating Your Application

To deploy updates:

```bash
git add .
git commit -m "Update description"
git push heroku main
```

## Cost Considerations

- **Free Tier**: Heroku no longer offers a free tier
- **Basic Dyno**: Starts at ~$7/month
- **Hobby Dyno**: Good for small applications
- **Standard Dyno**: For production applications

## Security Notes

1. **Service Account**: Your `service_account.json` is now stored as an environment variable
2. **Secrets**: Never commit sensitive data to your repository
3. **HTTPS**: Heroku automatically provides SSL certificates

## Next Steps

1. Set up a custom domain (optional)
2. Configure monitoring and alerts
3. Set up automatic deployments from GitHub
4. Scale your application as needed

## Support

- [Heroku Documentation](https://devcenter.heroku.com/)
- [Heroku Support](https://help.heroku.com/)
- [Flask on Heroku](https://devcenter.heroku.com/articles/python-gunicorn)

Your ChessLytics application should now be live on Heroku! 🎉 