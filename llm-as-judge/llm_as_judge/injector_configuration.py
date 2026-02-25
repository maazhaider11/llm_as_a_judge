from injector import singleton, provider, inject, Module

from llm_as_judge.api.products.services import ProductService
from llm_as_judge.api.evaluations.services import EvaluationService, IEvaluationService
from llm_as_judge.api.evaluations.kg_service import KGService
from issm_common_database_setup.mongo import BeanieDBClient
from issm_api_common.config.urls import config
from neo4j import GraphDatabase, Session


class InjectorConfiguration(Module):
    @singleton
    @provider
    @inject
    def provide_mongo_db_client(self) -> BeanieDBClient:
        return BeanieDBClient(config.mongo_database_conn_str)

    @singleton
    @provider
    @inject
    def provide_neo4j_session(self) -> Session:
        # Pull from config (which should pull from env)
        uri = getattr(config, "neo4j_uri", "bolt://neo4j:7687")
        user = getattr(config, "neo4j_user", "neo4j")
        password = getattr(config, "neo4j_password", "password")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return driver.session()

    @singleton
    @provider
    @inject
    def provide_kg_service(self, session: Session) -> KGService:
        return KGService(session)

    @singleton
    @provider
    @inject
    def provide_evaluation_service(self, kg_service: KGService) -> IEvaluationService:
        return EvaluationService(kg_service=kg_service)

    @singleton
    @provider
    @inject
    def provide_product_service(self) -> ProductService:
        return ProductService()
