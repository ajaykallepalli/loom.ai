# Setting Up Google Cloud Storage for Loom.AI Style Transfer

This guide will walk you through how to set up a Google Cloud service account with the necessary permissions to access Cloud Storage for the Loom.AI Style Transfer application.

## Step 1: Create a Google Cloud Project (if you don't have one already)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click on "New Project"
4. Enter a name for your project and click "Create"
5. Once created, make sure your new project is selected in the project dropdown

## Step 2: Create a Google Cloud Storage Buckets

1. In the Google Cloud Console, navigate to "Cloud Storage" > "Buckets"
2. Click "Create Bucket"
3. Create the following three buckets with default settings:
   - `user-uploads-style-transfer-lab` - for user content uploads
   - `style-images-style-transfer-lab` - for style image uploads
   - `stylized-results-style-transfer-lab` - for generated results

## Step 3: Create a Service Account

1. In the Google Cloud Console, navigate to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Enter a name (e.g., "loom-ai-storage-account") and description
4. Click "Create and Continue"
5. Assign the following roles:
   - Storage Admin (`roles/storage.admin`) - this gives full access to GCS
   - Or for more restrictive permissions:
     - Storage Object Admin (`roles/storage.objectAdmin`)
6. Click "Continue" and then "Done"

## Step 4: Generate a Service Account Key

1. Find your service account in the list
2. Click the three dots menu (⋮) for that service account
3. Select "Manage keys"
4. Click "Add Key" > "Create new key"
5. Select "JSON" format
6. Click "Create" - the key file will automatically download to your computer

## Step 5: Configure Streamlit Secrets for Local Development

1. Create or edit the file `~/.streamlit/secrets.toml`
2. Add your service account key details:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "your-private-key"
client_email = "your-service-account-email"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

3. Copy the contents of your downloaded JSON key file into the corresponding fields

## Step 6: Run the GCS Connection Test

1. Run the test script to verify your connection:
```
streamlit run src/test_gcs.py
```

2. If successful, you should see:
   - Authentication success message
   - Your project ID
   - List of available buckets
   - Status of the required buckets

## Step 7: Deploy to Streamlit Cloud

1. Push your code to GitHub (don't include your secrets file!)
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app"
4. Connect to your GitHub repository
5. In the "Advanced settings" section, add your secrets in TOML format (same format as in Step 5)
6. For the "Main file path", use `src/app.py`
7. Click "Deploy!"

## Step 8: Test the Deployed Application

1. Once deployed, verify that the application can:
   - Upload content and style images
   - Process the images
   - Display the results

If you encounter any authentication issues, check:
1. That you've entered the service account details correctly in the Streamlit secrets
2. That the service account has the correct permissions
3. That the buckets exist and are accessible by the service account

## Troubleshooting

### "Failed to initialize Google Cloud Storage client"
- Make sure your service account credentials are properly formatted in the secrets.toml file
- Ensure the service account has the necessary permissions

### "Access Denied" errors
- Check if the buckets exist and if the service account has access to them
- Verify that the bucket names match exactly what's in the code

### Local vs. Deployed Differences
- For local development, you can also use Application Default Credentials:
  ```
  gcloud auth application-default login
  ```
- This eliminates the need for explicit service account credentials when testing locally 