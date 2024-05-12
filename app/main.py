from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import enriched_content_proposal, delete_nlp_model, user_feedback, user_delete, list_linguistic_analysis, list_enriched_content, global_config, local_config, load_nlp_model, webmaster_content_proposal, list_webmaster_content

app = FastAPI(debug=True, root_path='/api/enriched-content-service')
#app = FastAPI(debug=True)


app.include_router(enriched_content_proposal.router, tags=["enriched_content_proposal"])
app.include_router(list_linguistic_analysis.router, tags=["list_linguistic_analysis"])
app.include_router(list_enriched_content.router, tags=["list_enriched_content"])
app.include_router(global_config.router, tags=["global_config"])
app.include_router(local_config.router, tags=["local_config"])
app.include_router(load_nlp_model.router, tags=["load_nlp_model"])
app.include_router(delete_nlp_model.router, tags=["delete_nlp_model"])
app.include_router(webmaster_content_proposal.router, tags=["webmaster_content_proposal"])
app.include_router(list_webmaster_content.router, tags=["list_webmaster_content"])
app.include_router(user_feedback.router, tags=["user_feedback"])
app.include_router(user_delete.router, tags=["user_delete"])

cors_address="46.137.91.119"
cors_port="8001"

origins = [
    "http://darwinhost.dyndns.org:8001",
    "http://darwin.eulotech.it:8001",
    "http://darwin.eulotech.it",
    "https://darwin.eulotech.it",
    "http://localhost:8001",
    f"http://{cors_address}:{cors_port}",
    f"https://{cors_address}",
    f"http://{cors_address}"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


