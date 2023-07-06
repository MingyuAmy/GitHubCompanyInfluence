#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# provide common util methods for the scripts

import requests
import json
import math
import hashlib
import os.path
import csv
from datetime import datetime

# Create token from github account, settings > developer settings > Personal access tokens > Fine-grained tokens
# token(ming yu): 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
_tokens = [
    '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',
    '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'
]
_token_prefix = 'Bearer '
_headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.113 Safari/537.36',
    'Authorization': _token_prefix + _tokens[0]}

_API = {
    'api_limit': 'https://api.github.com/rate_limit',
    'fetch_repo': 'https://api.github.com/orgs/{}/repos?per_page={}&page={}',
    'company_info': 'https://api.github.com/orgs/{}',
    'repo_contributor': 'https://api.github.com/repos/{}/{}/contributors?per_page={}'
}


# cache util methods
# ---------------------
def read_cache(cache_fullpath):
    with open(cache_fullpath, 'r') as fd:
        json_str = fd.read()
    return json_str


def write_cache(cache_fullpath, json_str):
    with open(cache_fullpath, 'w') as fd:
        fd.write(json_str)


def cache_path(key, filename=None):
    key = hashlib.md5(key.encode()).hexdigest()
    cache_fname = key + '.json' if filename is None else filename
    cache_fullpath = os.path.join('cache', cache_fname)
    return cache_fullpath


def create_cache_dir(filename='cache'):
    if not os.path.exists(filename):
        os.mkdir(filename)


def read_lines_util_blank(filename):
    res = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                break
            res.append(line)
    return res


# ---------------------
# global count for the api call
count, token_idx = 0, 0

def switch_token():
    global count, token_idx
    if count >= 4990:
        count, token_idx = 0, (token_idx + 1) % len(_tokens)
        _headers['Authorization'] = _token_prefix + _tokens[token_idx]
        print('[debug] Switch to the next token')
        return True
    return False

def get_data_from_url(url, use_cache=False):
    """获取html/json数据, 并且解码"""
    cache_fullpath = cache_path(url)
    if use_cache and os.path.exists(cache_fullpath):
        json_str = read_cache(cache_fullpath)
    else:
        if 'api_limit' not in url:
            global count
            count += 1
            switch_token()
        res = requests.get(url, headers=_headers)
        json_str = res.content.decode()
        if use_cache:
            write_cache(cache_fullpath, json_str)
    return json_str


def get_api_limit():
    json_str = get_data_from_url(_API['api_limit'], False)
    try:
        remaining = json.loads(json_str)['resources']['core']['remaining']
        print('Github api limit for this hour is:', remaining)
        remaining = int(remaining)
        global count
        if count == 0:
            count = 5000 - remaining
        if switch_token():
            remaining = get_api_limit()
        return remaining
    except (json.JSONDecodeError, KeyError) as e:
        print('Illegal token')
        return 0


def get_company_repos(company, repo_num):
    cache_fullpath = cache_path('repos-{}'.format(company))
    if os.path.exists(cache_fullpath):
        json_str = read_cache(cache_fullpath)
        return json.loads(json_str)

    per_page = 100
    page_count = math.ceil(repo_num / per_page)

    result = []
    for page in range(page_count):
        repo_url = _API['fetch_repo'].format(company, per_page, page + 1)
        json_str = get_data_from_url(repo_url)
        try:
            data = json.loads(json_str)
            result += data
            print('Get repo count:', len(result), 'from url:', repo_url)
        except (json.JSONDecodeError, KeyError) as e:
            print('Illegal error occurred when getting url:', repo_url)
    write_cache(cache_fullpath, json.dumps(result))
    return result


def get_top_language(repos):
    freq = {}
    for repo in repos:
        lang = repo['language']
        if lang not in freq:
            freq[lang] = 1
        else:
            freq[lang] += 1
    max_count, max_key = 0, None
    for key, val in freq.items():
        if val > max_count and key:
            max_count, max_key = val, key
    return max_key


# cache in high level, so skip cache this one
def get_repo_contributors(url):
    per_page = 100

    result = []
    page_no = 1
    while True:
        repo_url = url + '?per_page=100&page=' + str(page_no)
        json_str = get_data_from_url(repo_url)
        try:
            data = json.loads(json_str)
            result += data
        except (json.JSONDecodeError, KeyError) as e:
            print('Failed to get contributors from url:', repo_url)
            break
        print('Get contributors:', len(result), 'from url:', repo_url)
        if len(data) < 100:
            break
        page_no += 1

    return result


def parse_company_status(json_str):
    try:
        data = json.loads(json_str)
        return {'followers': data['followers'],
                'repositories': data['public_repos'],
                'name': data['name'],
                'description': data['description'],
                'location': data['location'],
                'created_at': data['created_at'][:10]
                }
    except (json.JSONDecodeError, KeyError) as e:
        return None


def get_company_status(company):
    url = _API['company_info'].format(company)
    json_str = get_data_from_url(url)
    data = parse_company_status(json_str)
    if data is None:
        print('{} not exists! Please check the company name!'.format(company))
    return data


def fetch_contributors_for_repos(repos):
    # also needs to do with pagination
    # e.g. https://api.github.com/repos/google/guava/contributors?per_page=100
    contributors = set()
    for repo in repos:
        items = get_repo_contributors(repo['contributors_url'])
        for item in items:
            if type(item) == dict and 'login' in item:
                contributors.add(item['login'])
            else:
                print('[error] Found illegal contributor:', item)
    print('Find', len(contributors), 'contributors')
    return contributors


def fetch_contributors_for_company(company):
    cache_fullpath = cache_path('contributors-{}'.format(company))
    if os.path.exists(cache_fullpath):
        json_str = read_cache(cache_fullpath)
        return set(json.loads(json_str))

    data = get_company_status(company)
    repo_count = int(data['repositories'])
    repos = get_company_repos(company, repo_count)
    result = fetch_contributors_for_repos(repos)

    write_cache(cache_fullpath, json.dumps(list(result)))
    return result


def get_common_contributors(company_1, company_2):
    contr1 = fetch_contributors_for_company(company_1)
    contr2 = fetch_contributors_for_company(company_2)
    # find the common one
    return contr1.intersection(contr2)


def get_time_str(timeObj, timeFormat='%y-%m-%d_%H_%M'):
    return datetime.strftime(timeObj, timeFormat)


def save_to_csv(filename, header, rows):
    """
    Save data to csv file
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        # first write header and use comma separated
        writer.writerow(header)
        # write data to csv file
        for row in rows:
            writer.writerow(row)


# --------- cluster releated --------- #

import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle, islice
import csv, random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Map of program language to index (start from 0)
lang_map, index_to_lang_map = {}, {}


def replace_country(country):
    # Map country to number: China 0, Usa 1, Europe 2, Other 3
    if country == 'China':
        return 0
    elif country == 'USA':
        return 1
    elif country == 'Europe':
        return 2
    else:
        return 3


def get_country_name(idx):
    if idx == 0:
        return 'China'
    elif idx == 1:
        return 'USA'
    elif idx == 2:
        return 'Europe'
    else:
        return 'Other'


def replace_top_lang(lang):
    global lang_map
    if lang not in lang_map:
        lang_map[lang] = len(lang_map) + 1
    return lang_map[lang]


def get_top_lang_from_index(index):
    if not index_to_lang_map:
        for lang, index in lang_map.items():
            index_to_lang_map[index] = lang
    return index_to_lang_map[index]


def read_csv_data(filename):
    # given a csv file fullpath, and return the names and data matrix
    names = []
    data = []
    with open(filename, 'r', encoding='UTF-8') as csvfile:
        rows = csv.reader(csvfile)
        rows = [x for x in rows]
        for row in rows[1:]:
            data.append([int(x) for x in row[1:6]] + [replace_country(row[6]), replace_top_lang(row[7])])
            names.append(row[0])
    return np.array(names), np.array(data)

# company, h-index, repo count, contributor count, location, top_language
# only read first 4 columns
def read_csv_data2(filename):
    # given a csv file fullpath, and return the names and data matrix
    names = []
    data = []
    with open(filename, 'r', encoding='UTF-8') as csvfile:
        rows = csv.reader(csvfile)
        rows = [x for x in rows]
        for row in rows[1:]:
            data.append([int(x) for x in row[1:4]])
            names.append(row[0])
    return np.array(names), np.array(data)


# company, h-index, repo count, contributor count, repo-diff, contributor diff
# only read first 4 columns
def read_csv_data3(filename):
    # given a csv file fullpath, and return the names and data matrix
    names = []
    data = []
    with open(filename, 'r', encoding='UTF-8') as csvfile:
        rows = csv.reader(csvfile)
        rows = [x for x in rows]
        for row in rows[1:]:
            data.append([int(x) for x in row[1:6]])
            names.append(row[0])
    return np.array(names), np.array(data)



def read_matrix(filename):
    # given a csv file fullpath, and return the header and data matrix
    header = None
    data = []
    with open(filename, 'r') as csvfile:
        rows = csv.reader(csvfile)
        rows = [x for x in rows]
        header = rows[0][1:]
        for row in rows[1:]:
            data.append([int(x) for x in row[1:]])
    return np.array(header), np.array(data)


def plot_and_save(X, y, out='no1.png', title='Kmeans Clustering', xlabel='h-index', ylabel='Repo number'):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y) + 1))))
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(out)
    print('Save graph to', out)


def plot3d_and_save(X, y, out='no1.png', title='Kmeans Clustering', xlabel='h-index', ylabel='Repo number',
                    zlabel='Company Size(1: small, 2: large)'):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y) + 1))))
    # print(colors)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(X)):
        color = colors[y[i]]
        xs, ys, zs = X[i, 0], X[i, 1], X[i, 2]
        ax.scatter(xs, ys, zs, color=color)

    # plt.scatter(X[:,0], X[:,1], s=10, color=colors[y])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # plt.show()
    plt.savefig(out)
    print('Save graph to', out)


def normalize(X):
    #  X_normalized = preprocessing.normalize(X, norm='l2')
    #  return X_normalized

    X = MinMaxScaler().fit_transform(X)
    # X = StandardScaler().fit_transform(X)
    return X


# --------- training releated --------- #


def evaluate_model(model, X, y, verbose=False):
    res = model.predict(X)
    if verbose:
        res = res.astype('int')
        print('the predict result')
        print(res, np.shape(res))

        print('the ground truth')
        print(y, np.shape(y))

    # The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
    score = model.score(X, y)
    print('model score: {:.3} (possible score range: [-1, 1])'.format(score))
    return score


def show_roc(y_test, y_pred_prob):
    from sklearn import metrics
    import matplotlib.pyplot as plt

    # ROC-AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='xbg estimator')
    display.plot()
    plt.show()