# Deployment Guide for Fertilizer Recommendation API

This guide provides multiple deployment options for your FastAPI backend.

## Option 1: Local Development (Recommended for Testing)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python run_server.py
   ```
   
   Or directly with uvicorn:
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

3. **Test the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## Option 2: Railway Deployment (Free Tier Available)

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway:**
   ```bash
   railway login
   ```

3. **Initialize project:**
   ```bash
   railway init
   ```

4. **Deploy:**
   ```bash
   railway up
   ```

5. **Get your public URL:**
   ```bash
   railway domain
   ```

## Option 3: Render Deployment (Free Tier Available)

1. **Create a Render account** at https://render.com

2. **Create a new Web Service**

3. **Connect your GitHub repository**

4. **Configure the service:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3.9+

5. **Deploy and get your public URL**

## Option 4: Heroku Deployment

1. **Install Heroku CLI**

2. **Create Procfile:**
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

3. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Option 5: Docker Deployment

1. **Build the image:**
   ```bash
   docker build -t fertilizer-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 fertilizer-api
   ```

## Environment Variables

Set these environment variables in your deployment platform:

- `PORT`: Port number (usually set automatically)
- `ML_DATASET_PATH`: Path to your dataset (optional)

## Testing Your Deployed API

1. **Health Check:**
   ```bash
   curl https://your-api-url.railway.app/health
   ```

2. **Get Prediction:**
   ```bash
   curl -X POST "https://your-api-url.railway.app/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "Temperature": 25.0,
          "Humidity": 80.0,
          "Moisture": 30.0,
          "Soil_Type": "Loamy",
          "Crop_Type": "rice",
          "Nitrogen": 85.0,
          "Potassium": 45.0,
          "Phosphorous": 35.0
        }'
   ```

## Frontend Integration

After deployment, update your frontend ML API service:

```typescript
// src/services/mlApiService.ts
constructor() {
  // Replace with your deployed API URL
  this.baseUrl = 'https://your-api-url.railway.app';
}
```

## Monitoring and Logs

- **Railway:** Use Railway dashboard for logs and monitoring
- **Render:** Check the logs tab in your service dashboard
- **Heroku:** Use `heroku logs --tail` for real-time logs

## Troubleshooting

### Common Issues:

1. **Port already in use:**
   - Change the port in your deployment configuration
   - Use `$PORT` environment variable for cloud platforms

2. **Model loading fails:**
   - Ensure the dataset path is correct
   - Check if all dependencies are installed

3. **CORS issues:**
   - The API already includes CORS middleware
   - If issues persist, check your frontend domain

4. **Memory issues:**
   - Reduce `n_estimators` in the RandomForest model
   - Use smaller dataset if available

## Performance Optimization

1. **Model Caching:** The model is loaded once on startup
2. **Async Processing:** All endpoints are async for better performance
3. **Input Validation:** Comprehensive validation prevents unnecessary processing
4. **Error Handling:** Graceful error handling with meaningful messages

## Security Considerations

1. **Input Validation:** All inputs are validated
2. **CORS Configuration:** Configure allowed origins for production
3. **Rate Limiting:** Consider adding rate limiting for production use
4. **Authentication:** Add authentication if needed for production

## Support

If you encounter issues:
1. Check the logs in your deployment platform
2. Test locally first with `python test_model.py`
3. Verify all dependencies are installed
4. Check the API documentation at `/docs` endpoint
