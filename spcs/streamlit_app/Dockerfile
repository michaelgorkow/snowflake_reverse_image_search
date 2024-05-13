FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip && pip install streamlit pandas snowflake-snowpark-python[pandas]==1.14.0 matplotlib Pillow scikit-image scikit-learn

COPY streamlit_app.py /app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]