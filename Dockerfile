FROM tensorflow/serving:latest-gpu
COPY models /models
EXPOSE 8501
ENTRYPOINT ["tensorflow_model_server",
            "--model_config_file=/models/models.config",
            "--allow_version_labels_for_unavailable_models=true",
            "--rest_api_port=8501"]
