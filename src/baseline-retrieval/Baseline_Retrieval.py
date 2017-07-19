import gensim

train_triple_name = "./train-webnlg-all-delex.triple"
train_lex_name = "./train-webnlg-all-delex.lex"
test_file_name = "./dev-webnlg-all-delex.triple"

if __name__ == '__main__':
    train_file_input = open(train_triple_name, "r").read().split('\n')
    train_lex_input = open(train_lex_name, "r").read().split('\n')
    texts = [i.split(' ') for i in train_file_input]
    bow_dictionary = gensim.corpora.Dictionary(texts)
    corpus = [bow_dictionary.doc2bow(text) for text in texts]
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    index_sim_tf_idf = gensim.similarities.Similarity("./", corpus_tfidf, len(bow_dictionary) )
    #index_sim_tf_idf.save('./index_sim_tf_idf.index')
    test_file_input = open(test_file_name, 'r').read().split('\n')
    out_str = ''
    for j in test_file_input:
        vec_bow = bow_dictionary.doc2bow(j.split(' '))
        vec_tfidf = tfidf[vec_bow]
        sims_lsi = index_sim_tf_idf[vec_tfidf]
        sims_lsi = sorted(enumerate(sims_lsi), key = lambda item: -item[1])
        out_str += train_lex_input[sims_lsi[0][0]] + '\n'
    open("./baseline_predictions.txt", "w").write(out_str)