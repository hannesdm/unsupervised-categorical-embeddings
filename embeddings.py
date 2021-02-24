# !pip install -U git+https://github.com/hannesdm/glove-python.git

import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from glove import Corpus, Glove
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import os



def create_vocab_dict(df):
    col_uniques = []
    for c in df.columns:
        col_uniques.append(df[c].unique())
    word_id_dict = {}
    cntr = 0
    for col in col_uniques:
        for word in col:
            word_id_dict[word] = cntr
            cntr = cntr + 1
    return word_id_dict


def glove_vectors(x, embedding_size, epochs=50, lr=0.05, alpha=0.75, max_count=100,tmp_loc='glove.w2vmodel'):
    # create dict ourselves so that the ids correspond to their location in the df, starting to count from first col downwards
    df = pd.DataFrame(x)
    word_id_dict = create_vocab_dict(df)
    # Creating a corpus object
    corpus = Corpus(dictionary=word_id_dict)
    # Training the corpus to generate the co occurence matrix which is used in GloVe
    # Distance scaling: standard glove reduces the occurence count based on how far a context word is from the focus word.
    # Should not be used since distance has no meaning for purely categorical variables.
    corpus.fit(df.values.tolist(), window=len(df.columns), distance_scaling=False)
    # alpha is the weighing of the loss, based on how likely a cooccurence is (Xij), less likely = less weight.
    glove = Glove(no_components=embedding_size, learning_rate=lr, alpha=alpha, max_count=max_count)
    glove.fit(corpus.matrix, epochs=epochs, no_threads=1,verbose=True)  # glove paper: 50 epochs for dimensionality <300, 100 otherwise
    glove.add_dictionary(corpus.dictionary)
    glove.save_word2vec_format(tmp_loc)

    model = KeyedVectors.load_word2vec_format(tmp_loc)
    if os.path.exists(tmp_loc):
        os.remove(tmp_loc)
    return model


def w2vec_vectors(x, embedding_size=100, sg=0, negative=10, min_count=1):
    if type(x) is pd.DataFrame:
        x = x.values
    model = gensim.models.Word2Vec(x.tolist(), min_count=min_count, size=embedding_size, sg=sg, window=x.shape[1], negative=negative)  # sg = skigram, 0 means cbow
    # negative sampling often performs better, 10 is used in glove paper
    return model


def lsa(df, memmap=False, n_components=10):
    unique_words = []
    unique_words_dict = {}
    cntr = 0
    for c in df.columns:
        unique_col_words = df[c].unique()
        unique_words.extend(df[c].unique())
        for w in unique_col_words:
            unique_words_dict[w] = cntr
            cntr = cntr + 1

    all_vals = []
    for idx, row in df.iterrows():
        vals = [1 if x in row.values else 0 for x in unique_words]
        all_vals.append(vals)
    binarized = np.array(all_vals)

    svd = TruncatedSVD(n_components)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    vectors = lsa.fit_transform(binarized.T)

    if memmap:
        x_embedded = np.memmap("x.np", dtype='float32', mode='w+', shape=(df.shape[0], df.shape[1], n_components))
    else:
        x_embedded = np.zeros((df.shape[0], df.shape[1], n_components))
    x = df.values
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_embedded[i, j, :] = vectors[unique_words_dict[x[i, j]]]
    x_embedded = x_embedded.reshape(x_embedded.shape[0], -1)

    return x_embedded, vectors


def embed_data(x, model, embedding_size, memmap=False, memmap_name="x.np"):
    model.wv.init_sims()  # calculate normalized vectors
    if memmap:
        x_embedded = np.memmap(memmap_name, dtype='float32', mode='w+', shape=(x.shape[0], x.shape[1], embedding_size))
    else:
        x_embedded = np.zeros((x.shape[0], x.shape[1], embedding_size))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_embedded[i, j, :] = model.wv.word_vec(x[i, j], use_norm=True)
    x_embedded = x_embedded.reshape(x_embedded.shape[0], -1)
    return x_embedded


def embed_w2v(x, embedding_size=100, sg=0, memmap=False, memmap_name="x.np"):
    model = w2vec_vectors(x, embedding_size, sg)
    return embed_data(x, model, embedding_size, memmap, memmap_name)


def embed_glove(x, embedding_size=100, epochs=50, lr=0.05, alpha=0.75, max_count=100, memmap=False,
                memmap_name="x.np"):
    model = glove_vectors(x, embedding_size, epochs=epochs, lr=lr, alpha=alpha, max_count=max_count)
    return embed_data(x, model, embedding_size, memmap, memmap_name)


def plot_tsne(model,df, words=None, vectors=None, target_subfeatures_mask=None, perplexity=5, title="TSNE", colour = False, n_components=2):

    if model is not None:
        words = []
        vectors = []
        for word in model.wv.vocab.keys():
            words.append(word)
            vectors.append(model.wv.word_vec(word))

    if colour:
      cols = df.columns
      cmap = plt.get_cmap('viridis')
      clrs = cmap(np.linspace(0, 1, len(cols)))
      clr_dict = {}
      clrs_points = []
      for i,c in enumerate(cols):
        unique_col_vals = df[c].unique()
        for uv in unique_col_vals:
          clr_dict[uv] = clrs[i]
      for w in words:
        clrs_points.append(clr_dict[w])

    tsne = manifold.TSNE(n_components=n_components, init='pca'
      ,random_state=10,method ='exact',perplexity = perplexity)
    Y = tsne.fit_transform(vectors)

    plt.figure(figsize=(18,12))
    if target_subfeatures_mask is None:
        target_subfeatures_mask = np.array([False] * Y.shape[0])
    plt.rcParams.update({'font.size': 14}) # set everything to this font size
    marker_size = plt.rcParams['lines.markersize'] ** 2 * 5
    if colour:
      plt.scatter(Y[~target_subfeatures_mask,0],Y[~target_subfeatures_mask,1], c = clrs_points, s= marker_size)
    else:
      plt.scatter(Y[target_subfeatures_mask,0],Y[target_subfeatures_mask,1], c = 'red', s= marker_size)
      plt.scatter(Y[~target_subfeatures_mask,0],Y[~target_subfeatures_mask,1], c = 'blue', s= marker_size)

    for i, (label) in enumerate(words):
        plt.annotate(label, (Y[i,0],Y[i,1]), fontsize=22)

    plt.title(title, fontsize=26)
    plt.xlabel("Dimension 1", fontsize=26)
    plt.ylabel("Dimension 2", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.savefig("tsne_2_components")
    plt.show()


def plot_pca(model,df, words=None, vectors=None, target_subfeatures_mask=None, title="PCA", colour = False,n_components=2):
    if model is not None:
        words = []
        vectors = []
        for word in model.wv.vocab.keys():
            words.append(word)
            vectors.append(model.wv.word_vec(word))

#     colours for plots - each column gets a color
    if colour:
      cols = df.columns
      cmap = plt.get_cmap('viridis')
      clrs = cmap(np.linspace(0, 1, len(cols)))
      clr_dict = {}
      clrs_points = []
      for i,c in enumerate(cols):
        unique_col_vals = df[c].unique()
        for uv in unique_col_vals:
          clr_dict[uv] = clrs[i]
      for w in words:
        clrs_points.append(clr_dict[w])
    pca = PCA(n_components=n_components)
    Y = pca.fit_transform(vectors)
    plt.figure(figsize=(18,12))
    if target_subfeatures_mask is None:
        target_subfeatures_mask = np.array([False] * Y.shape[0])
    plt.rcParams.update({'font.size': 14}) # set everything to this font size
    marker_size = plt.rcParams['lines.markersize'] ** 2 * 5
    if colour:
      plt.scatter(Y[~target_subfeatures_mask,0],Y[~target_subfeatures_mask,1], c = clrs_points, s= marker_size)
    else:
      plt.scatter(Y[target_subfeatures_mask,0],Y[target_subfeatures_mask,1], c = 'red', s= marker_size)
      plt.scatter(Y[~target_subfeatures_mask,0],Y[~target_subfeatures_mask,1], c = 'blue', s= marker_size)

    for i, (label) in enumerate(words):
        plt.annotate(label, (Y[i,0],Y[i,1]), fontsize=22)

    plt.title(title, fontsize=26)
    plt.xlabel("Dimension 1", fontsize=26)
    plt.ylabel("Dimension 2", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.savefig("pca_2_components")
    plt.show()