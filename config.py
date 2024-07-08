class Config:
    PAGE_TITLE = "Image Analyser"

    ASTRA_VECTOR_ENDPOINT = "https://8687c3e7-66c5-4679-9c59-b405038cfec2-us-east1.apps.astra.datastax.com"
    ASTRA_DB_KEYSPACE = "imagesearch"
    ASTRA_DB_COLLECTION = "images"

    EMBEDDING_MODEL = "text-embedding-3-small"
    #VISION_MODEL = "gpt-4-turbo"
    VISION_MODEL = "gpt-4o"
    MAX_TOKENS = 1000
    DETAIL = "high"

    TOP_K_VECTORSTORE = 3
    TOP_K_MEMORY = 1

    INITIAL_SIDEBAR_STATE = "collapsed"
    #INITIAL_SIDEBAR_STATE = "expanded"