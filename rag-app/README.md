# RAG-APP

To bring the service up:
```
oc apply -f dsr_model_serving.yml
oc apply -f dsr_milvus.yml
oc apply -f dsr_db_ui.yml
oc apply -f dsr_backend_server.yml
oc apply -f dsr_rag_ui.yml
```

To bring the service down:
```
oc delete deployment db-ui model-serve rag-backend rag-milvus rag-ui
oc delete service db-ui-service model-serve-svc rag-backend-service rag-milvus-svc rag-ui-service
oc delete route chat-route db-ui-route embed-route rag-backend-route rag-ui-route rerank-route vision-route
```
