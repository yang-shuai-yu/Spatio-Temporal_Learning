import torch
import pickle, os, random
import numpy as np
import argparse
import traj_dist.distance as tdist

def data_loader(args):
    with open('data/{}/gps/valgps'.format(args.model_name), 'rb') as f:
        querydata = np.array(pickle.load(f))
        f.close()
    with open('data/{}/gps/testgps'.format(args.model_name), 'rb') as f:
        database = np.array(pickle.load(f))
        f.close()

    with open('data/{}/{}_{}_val'.format(args.model_name, args.model_name, args.emb_dim), 'rb') as f:
        queryembs = np.array(pickle.load(f))
        f.close()
    with open('data/{}/{}_{}_test'.format(args.model_name, args.model_name, args.emb_dim), 'rb') as f:
        embsbase = np.array(pickle.load(f))
        f.close()

    return querydata, database, queryembs, embsbase

def dist_compute(query, database):
    counter = 0
    dists = np.zeros((len(query), len(database)))
    for i in range(len(query)):
        for j in range(len(database)):
            dists[i][j] = tdist.sspd(np.array(query[i]), np.array(database[j]))
            counter += 1
            if counter % 1000 == 0:
                print('Counter: {}'.format(counter))
    return dists

def true_distance_ranking(query, database):    # gps
    dists = np.zeros((len(query), len(database)))
    dists = dist_compute(query, database)
    topk_idx = np.argsort(dists, axis=1)    # shape: (query, database)
    return topk_idx

def topk_ranking(queryembs, embsbase, topk):
    # calculate the distance between query and database using cosine similarity
    # dists = np.dot(queryembs, embsbase.T)

    nums = np.dot(queryembs, embsbase.T)
    denom = np.linalg.norm(queryembs, axis=1).reshape(-1, 1) * np.linalg.norm(embsbase, axis=1)  # 求模长的乘积
    dists = nums / denom
    
    # get the topk ranking
    topk_idx = np.argsort(dists, axis=1)[:, -topk:]
    return topk_idx

def evaluate_performance(topk_sim_idx, topk_true_idx, args):
    # --- top_k similarity rank of true distance rank
    # fine the topk_sim_idx in topk_true_idx
    # calculate the ranking
    topk_sim_rank = np.zeros(topk_sim_idx.shape)
    for i in range(topk_sim_idx.shape[0]):
        for j in range(topk_sim_idx.shape[1]):
            topk_sim_rank[i][j] = np.where(topk_true_idx[i] == topk_sim_idx[i][j])[0][0]
    print('Topk similarity rank: {}'.format(topk_sim_rank))
    print('Mean topk similarity rank: {}'.format(np.mean(topk_sim_rank)))

    # --- calculate the accuracy
    accuracy = np.zeros(topk_sim_idx.shape[0])
    for i in range(topk_sim_idx.shape[0]):
        # print(np.sum(topk_sim_rank[i] <= args.topk-1) / args.topk)
        accuracy[i] = np.sum(topk_sim_rank[i] <= args.topk-1) / args.topk
    print('Accuracy: {}'.format(accuracy))

    # save the performance
    save_performance(accuracy, topk_sim_rank, args)

def save_performance(accuracy, topk_sim_rank, args):
    with open(os.path.join(args.save_dir, args.model_name+'{}_performance.txt'.format(args.emb_dim)), 'w') as f:
        f.write('Accuracy: {}\n'.format(accuracy))
        f.write('Mean accuracy: {}\n'.format(np.mean(accuracy)))
        f.write('Topk similarity rank: {}\n'.format(topk_sim_rank))
        f.write('Mean topk similarity rank: {}\n'.format(np.mean(topk_sim_rank)))
        f.write('Mean topk similarity rank / topk: {}\n'.format(np.mean(topk_sim_rank) / args.topk))
        f.close()


if __name__ == '__main__':
    # load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='neutraj')
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--num_query', type=int, default=100)
    parser.add_argument('--num_database', type=int, default=1000)
    args = parser.parse_args()
    
    # Load the data
    # -- Load the gps data, take valid data as query, and test data as database
    # -- Load the embeddings as query and database
    querydata, database, queryembs, embsbase = data_loader(args)

    # Evaluate the performance by ranking
    # select 100 queries, and evaluate the top 1 ranking
    # get the distance between query and database and calculate the topk true-distance ranking
    num_query = args.num_query
    topk = args.topk
    select_query_idx = random.sample(range(len(querydata)), num_query)
    select_database_idx = random.sample(range(len(database)), args.num_database)
    topk_sim_idx = topk_ranking(queryembs[select_query_idx], embsbase[select_database_idx], topk)    # get the topk ranking by similarity
    print(topk_sim_idx.shape)
    topk_true_idx = true_distance_ranking(querydata[select_query_idx], database[select_database_idx])    # get the ranking by true distance
    print(topk_true_idx.shape)

    # calculate the performance and save the results
    evaluate_performance(topk_sim_idx, topk_true_idx, args)



    


              