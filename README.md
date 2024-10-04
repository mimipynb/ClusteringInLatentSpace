# Latent Space Clustering

Queries are encoded using pre-trained language models, and their embeddings are mean-pooled to retain contextual information. Leveraging this preserved contextual information, a clustering model can be applied to group similar queries (or corpuses), where the similarity is measured by the projected distances between the query embeddings.

#### Applications of this method:
- Personalizing the embeddings based on new user behavior by representing queries (or the corpus) with contextual information tailored to the user’s preferences and usage patterns.  
- Can be used to automatically group semantically similar queries, improving search and retrieval systems.
- Fine-tuning pre-trained binary/multi-label classifiers without the need to fine-tuning the model as a whole.
- Anomaly detection by identifying queries or data points that deviate significantly from the clustered embeddings in the latent space (ie: the differences between the distributions of the unseen data against previously seen data can be measured via distance methods within L2 or via KL divergence (although this may be less efficient))

#### Extensions:
Purpose of this experiment is to personalize pre-trained models while avoiding the need to fine-tune the models. Other backlogged tasks for this method:
- Fine-tune pre-trained multi-label classifiers (eg: https://huggingface.co/SamLowe/roberta-base-go_emotions).
- Test the feasibility of constructing a Self-Organizing Map (SOM) using this method.

#### Doubts:
- Based on past experience, training a layer using distance-based loss functions has not yielded promising results as the model learns from more data. Hence, this might need further investigation's on the training params and network's architecture (eg: dimensions). If the previous two approaches prove to be less trivial as I'd hope, I aim to estimate the timeframe during which the embeddings remain applicable, or explore model integration techniques to effectively pool the embeddings together.

#### Used Packages:
- torch
- sentence_transformer
- transformers
