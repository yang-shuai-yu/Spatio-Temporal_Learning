import json


def info_write(dataset):
    """
    Write info for a dataset
    """
    info = {
        "dataset": "{}".format(dataset),
        "maxlat": 41.307945,
        "minlat": 40.953673,
        "maxlon": -8.156309,
        "minlon": -8.735152,
    }
    
    js = json.dumps(info)   
    file = open('./data/trajs/{}/info.txt'.format(dataset), 'w')
    file.write(js)
    file.close()

if __name__ == '__main__':
    info_write("porto")