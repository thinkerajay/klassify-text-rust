# klassify-text-rust
A Rest Service to classify text using gzip and KNN

References:
- Less is More: Parameter-Free Text Classification with Gzip(https://arxiv.org/pdf/2212.09410.pdf)
- Tsoding Daily

### Implementation
- A http server using actix-web with a single endpoint accepting POST request with a payload.
- Preload training data (tested with AG news dataset) before starting the server.
- Store klass, original_text, compressed_text as global Vec.
- Used K = 1000 for testing 
