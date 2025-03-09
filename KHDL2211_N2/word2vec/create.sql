CREATE DATABASE steamdataset IF NOT EXISTS;

CREATE TABLE glove_25 IF NOT EXISTS (
    id bigserial PRIMARY KEY, embedding vector(25), document TEXT
);