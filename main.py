import jieba
import os
import re
from urllib.request import urlretrieve

import jieba
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import psycopg2



def is_2letters(word):
    return True if re.match(r"^[a-zA-Z]{1,2}$", word) else False


def is_number(word):
    return True if re.match(r"^[0-9]+$", word) else False


def remove_stop_word(words):
    stop_words = set()
    with open("stop_words.txt", 'r', encoding='utf-8') as f:
        for w in f.readlines():
            stop_words.add(w.replace("\n", ""))
    new_words = [word for word in words if
                 word not in stop_words and len(word) > 1 and not is_number(word) and not is_2letters(
                     word) and word != "\r\n"]
    return new_words


def add_big_dic():
    # 使用大型字典
    big_dict_path = "dict.txt.big"
    if not os.path.exists(big_dict_path):
        print("下載大型字典")
        url = "https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big"
        urlretrieve(url, big_dict_path)
    jieba.set_dictionary(big_dict_path)


def alibaba(text):
    chars = [char for char in text if char]
    return chars


def cut_head_and_tail(words, head_percentage, art_window):
    start_index = int(len(words) * head_percentage)
    end_index = start_index + art_window
    result = words[start_index: end_index]
    return result


def read_file(file):
    text = None
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def train(is_male):
    gender = 'male' if is_male else 'female'

    add_big_dic()
    csv_file_name = gender + '.csv'
    df_train = pd.read_csv(csv_file_name)
    print(csv_file_name, 'read finish')
    df_train2 = pd.read_csv("common.csv")
    print('common.csv read finish')
    products = pd.concat([df_train, df_train2], axis=0)
    product_dict = {str(r['id']): r['content'] for i, r in products.iterrows()}
    print('product_dict generated')
    product_clean_dict = {k: remove_stop_word(jieba.lcut(str(product_dict[k]))) for k in product_dict}
    documents = [TaggedDocument(product_clean_dict[k], [k]) for k in product_clean_dict]
    # print(documents[:10])
    # return
    model = Doc2Vec(documents, dm=1, dm_mean=1, vector_size=512,
                    window=12, min_count=5, alpha=0.025,
                    workers=6, epochs=15, dbow_words=1, seed=0)
    model.save('article_' + gender + 'Ncommon.model')
    print("Done")
    data = load_model_and_training_data(is_male)
    return data


def get_pgconn():
    conn = psycopg2.connect(user="Alex", password="alexalex", database="gift_expert_testenv", host="172.104.89.11",
                            port="5432")
    return conn


#
def get_article_about_bob(bob_info, is_male, topn):
    model = load_article_model(is_male)
    result = dict()
    result['article_words'] = remove_stop_word(jieba.lcut(bob_info))
    result['article_vec'] = model.infer_vector(result['article_words'])
    sim = model.docvecs.most_similar([result['article_vec']], topn=topn)
    result['article_id_list'] = [int(article_id) for (article_id, rank) in sim]
    result['bob_info'] = bob_info
    return result

def create_article_sugeest_product():
    # True runs every procedure then False after it.
    conn = get_pgconn()
    for is_male in [True, False]:
        gender = '男生' if is_male else '女生'
        model = load_product_model(is_male)
        sql = "select id, compact_content from articles where gender_type = %s or gender_type is null " \
              "order by id limit "
        with conn.cursor() as cur:
            cur.execute(sql, (gender,))
            article_list = cur.fetchmany(1000)
            print("load 1000 records")
            while len(article_list) > 0:
                for article in article_list:
                    article_id, article_text = article[0], article[1]
                    suggest_data = suggest_gift_by_article(article_text, model, 10, .15, 200)
                    print("gender_type:", gender, "article_id:", article_id, "product_id_list:",
                          suggest_data["product_id_list"])
                    save_article_prodcuts(conn, article_id, suggest_data["product_id_list"], is_male)
                    conn.commit()
                    print("saved and committed")
                article_list = cur.fetchmany(1000)
                print("load 1000 records")


def save_article_prodcuts(conn, article_id, product_id_list, is_male):
    gender = '男生' if is_male else '女生'
    sql = 'insert into article_products (article_id, product_id, gender_type) values (%s, %s, %s) ' \
          'on conflict(article_id, product_id, gender_type) do NOTHING'
    cur = conn.cursor()
    for product_id in product_id_list:
        cur.execute(sql, (article_id, product_id, gender))
    conn.commit()


def load_model_and_training_data(is_male):
    gender = 'male' if is_male else 'female'
    result = dict()
    products = list(pd.read_csv("products_202009161312.csv")['description'])
    articles = list(pd.read_csv("articles_202009161311.csv")['compact_content'])
    model = Doc2Vec.load('article_' + gender + 'Ncommon.model')
    model.random.seed(0)
    result['articles'] = articles
    result['products'] = products
    result['model'] = model
    return result

def get_suggestion_for_alice(bob_info, is_male, tag_list, conn):
    return {"product_info": suggest_gift_by_bob_info(bob_info, is_male, tag_list, conn), "product_tag_info":  suggest_gift_tags_by_bob_info(bob_info, is_male, tag_list, conn)}

def suggest_gift_by_bob_info(bob_info, is_male, tag_list, conn):
    article_id_list = list(map(str, get_article_about_bob(bob_info, is_male, 100)['article_id_list']))
    article_id_list_str = ", ".join(article_id_list)
    cur = conn.cursor()
    gender = '男生' if is_male else '女生'
    tag_list_str = "'" + "', '".join(tag_list) + "'"
    selected_phrase = ', 1 as selected_sum' if len(tag_list)==0 else f", sum(case when  pp.tag_name in ({tag_list_str}) then 1 else 0 end ) as selected_sum"
    sql = f"""select product_id, p."name", p.description
from (
	select b.product_id as product_id, b.selected_sum
	from(
		select pp.product_id {selected_phrase}
		from (
			select distinct(product_id)
			from article_products ap
			where ap.article_id in ({article_id_list_str}) and ap.gender_type = %s 
		) as a inner join product_ptags pp on a.product_id = pp.product_id 
		group by pp.product_id 
	) as b
	where b.selected_sum >= %s
) as c inner join products p on p.id = c.product_id 
"""

    cur.execute(sql, (gender, len(tag_list)))
    products = cur.fetchall()
    return products

def suggest_gift_tags_by_bob_info(bob_info, is_male, tag_list, conn):
    article_id_list = list(map(str, get_article_about_bob(bob_info, is_male, 100)['article_id_list']))
    article_id_list_str = ", ".join(article_id_list)
    cur = conn.cursor()
    gender = '男生' if is_male else '女生'
    tag_list_str = "'" + "', '".join(tag_list) + "'"
    selected_phrase = '1 as selected_sum' if len(tag_list)==0 else f"sum(case when  pp.tag_name in ({tag_list_str}) then 1 else 0 end ) as selected_sum"
    where_phrase = '' if len(tag_list)==0 else f"where pp2.tag_name not in ({tag_list_str})"

    sql = f"""select pp2.tag_name, count(pp2.tag_name) as cnt
from (
	select b.product_id, b.selected_sum
	from(
		select pp.product_id , {selected_phrase}
		from (
			select distinct(product_id)
			from article_products ap
			where ap.article_id in ({article_id_list_str}) and ap.gender_type = %s 
		) as a inner join product_ptags pp on a.product_id = pp.product_id 
		group by pp.product_id 
	) as b
	where selected_sum >= %s
) as c inner join product_ptags pp2 on pp2.product_id =c.product_id
{where_phrase}
group by pp2.tag_name
order by cnt desc 
"""
    print(sql)

    cur.execute(sql, (gender, len(tag_list)))
    product_tags = cur.fetchall()
    return product_tags




def load_product_model(is_male):
    gender = 'male' if is_male else 'female'
    model = Doc2Vec.load('product_' + gender + 'Ncommon.model')
    return model


def load_article_model(is_male):
    gender = 'male' if is_male else 'female'
    model = Doc2Vec.load('article_' + gender + 'Ncommon.model')
    return model


def suggest_gift_by_article(text, model, topn, start_percentage, art_window) -> dict:
    result = dict()
    result['article_words'] = cut_head_and_tail(remove_stop_word(jieba.lcut(text)), start_percentage, art_window)
    result['article_vec'] = model.infer_vector(result['article_words'])
    sim = model.docvecs.most_similar([result['article_vec']], topn=topn)
    print([product_id for (product_id, rank) in sim])
    result['product_id_list'] = [int(product_id) for (product_id, rank) in sim]
    result['article'] = text
    return result

# input:BOB腦袋
def tell_me_why(source_text, target_text, max_depth, wv, topn, start_percentage, art_window):
    clean_words_short = cut_head_and_tail(remove_stop_word(jieba.lcut(source_text)), start_percentage, art_window)
    source_word_set = set(model_view(wv, " ".join(clean_words_short)))
    target_word_set = set(model_view(wv, target_text))
    print("source:", source_text, "\n", source_word_set)
    print("target:", target_text, "\n", target_word_set)
    reason = []
    for s in source_word_set:
        paths = search_paths(s, target_word_set, max_depth, wv, topn)
        reason += paths
    return reason


def search_paths(source_word, target_word_set, limit_depth, wv, topn):
    paths = []
    temp_paths = [[source_word]]
    i = 0
    while True:
        cur_path = temp_paths[i]
        if len(cur_path) == limit_depth:
            break
        next_words = get_similar_words(cur_path[-1], wv, topn)
        for n in next_words:
            new_path = cur_path + [n]
            temp_paths.append(new_path)
            if n in target_word_set:
                paths.append(new_path)
                print(source_word, "found:", new_path)
        i += 1
    return paths


def get_similar_words(word, wv_model, topn):
    words = [w for w, _ in wv_model.most_similar([word], topn=topn)]
    return words


def model_view(model, text):
    after_stop_text_words = remove_stop_word(jieba.lcut(text))
    model_view_words = [w for w in after_stop_text_words if w in model.wv.vocab]
    return model_view_words


def get_model_view_set(model, text):
    return set(model_view(model, text))


def main():
    train(False)
    # predict()
    # suggest_gift()


if __name__ == '__main__':
    main()
