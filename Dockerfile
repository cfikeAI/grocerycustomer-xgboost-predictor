#python image
FROM python:3.10-slim

#Set working directory
WORKDIR /app

#Copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy entire project into container
COPY . .    

#Expose port for FastAPI
EXPOSE 8000

#Run the app with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]




