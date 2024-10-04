# Latent Space Clustering

Queries are encoded using pre-trained language models, and their embeddings are mean-pooled to retain contextual information. Leveraging this preserved contextual information, a clustering model can be applied to group similar queries (or corpuses), where the similarity is measured by the projected distances between the query embeddings.

#### Applications of this method:
- Personalizing the embeddings based on new user behavior by representing queries (or the corpus) with contextual information tailored to the user’s preferences and usage patterns.  
- Can be used to automatically group semantically similar queries, improving search and retrieval systems.
- Fine-tuning pre-trained binary/multi-label classifiers without the need to fine-tuning the model as a whole.
- Anomaly detection by identifying queries or data points that deviate significantly from the clustered embeddings in the latent space (ie: the differences between the distributions of the unseen data against previously seen data can be measured via distance methods within L2 or via KL divergence (although this may be less efficient))

#### Extensions:
- Fine-tune pre-trained multi-label classifiers like Roberta https://huggingface.co/SamLowe/roberta-base-go_emotions
- Test whether a Self-organizing map can be constructed from this method.

#### Used Packages:
- torch
- sentence_transformer
- transformers
