from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec
from torch.nn.init import sparse


def create_indexes(pc, dense_index_name, sparse_index_name):
    if not pc.has_index(dense_index_name):
        dense_index_model = pc.create_index(
            name=dense_index_name,
            vector_type="dense",
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
            tags={"environment":"development"}
        )

    if not pc.has_index(sparse_index_name):
        sparse_index_model = pc.create_index(
            name=sparse_index_name,
            vector_type="sparse",
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
            tags={"environment":"development"}
        )


def dense_index_upsert(pc, dense_index_name, vectors):
    dense_index_host = pc.describe_index(dense_index_name).host
    dense_index = pc.Index(host=dense_index_host, grpc_config=GRPCClientConfig(secure=False))

    dense_index.upsert()

def sparse_index_upsert(pc, sparse_index_name, vectors):
    sparse_index_host = pc.describe_index(sparse_index_name).host
    sparse_index = pc.Index(host=sparse_index_host, grpc_config=GRPCClientConfig(secure=False))

    sparse_index.upsert()

def dense_index_query(pc, dense_index_name, q):
    dense_index_host = pc.describe_index(dense_index_name).host
    dense_index = pc.Index(host=dense_index_host, grpc_config=GRPCClientConfig(secure=False))

    dense_index.query()

def sparse_index_query(pc, sparse_index_name, q):
    sparse_index_host = pc.describe_index(sparse_index_name).host
    sparse_index = pc.Index(host=sparse_index_host, grpc_config=GRPCClientConfig(secure=False))

    sparse_index.query()

