import requests
import json
from initial import compute_embedding, image_preprocessing, read_images
products = {"0101": "https://www.pbs.org/wnet/nature/files/2014/10/Monkey-Main-1280x600.jpg"}
products_list = [
  {
    "_id": "6564d65505f0076b1d36576c",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104831_W_ms_DSC00936-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d36576e",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104830_ms_b_DSC00969-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365770",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104263_ms_a_DSC01018-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365772",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104418_ms_DSC00994-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365774",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104270_0002_DSC01062-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365776",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104389_DSC00849-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365778",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104929_ms_a_DSC00882-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d36577a",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104935_ms_a_DSC00791-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d36577c",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/09/0008_DSC06349-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d36577e",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/09/0019_P2_104213_DSC07710-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365780",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/07/0004_DSC03115__103720t.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365782",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/08/0015_P3_103577_DSC04578-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365784",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/08/0024_P2_103585_DSC04566-Edit.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365786",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/07/0005_DSC03056-Edit_102293t_103283p_103568b.jpg"
  },
  {
    "_id": "6564d65505f0076b1d365788",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/04/99295_ms.jpg"
  },
  {
    "_id": "6564d65505f0076b1d36578a",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2022/10/95534.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc501",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/11/100011_DSC09251-Edit_ms.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc503",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/01/100139_DSC00787_edited-new_ms.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc505",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/11/DSC09197-Edit_104824_ms.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc507",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/11/104203_DSC09079.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc509",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/11/104993_DSC09031-Edit.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc50b",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/11/100010_DSC08924-Edit.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc50d",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/11/104985_DSC08867-Edit.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc50f",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2022/10/95218_DSC08989-Edit.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc511",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104269_DSC09123-Editb.jpg"
  },
  {
    "_id": "6564d80647fa5358622dc513",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/100721-104281_new_ms.jpg"
  },
  {
    "_id": "6564d80747fa5358622dc515",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0082_104266_DSC01564-Edit.jpg"
  },
  {
    "_id": "6564d80747fa5358622dc517",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0059_p9_104317_DSC01728.jpg"
  },
  {
    "_id": "6564d80747fa5358622dc519",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0041_p12_104267_DSC01753.jpg"
  },
  {
    "_id": "6564d80747fa5358622dc51b",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0019_p16_104325_DSC01784.jpg"
  },
  {
    "_id": "6564d80747fa5358622dc51d",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104261_DSC09009.jpg"
  },
  {
    "_id": "6564d80747fa5358622dc51f",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104262_DSC09048-Edit.jpg"
  },
  {
    "_id": "6564d86ac2d5a4eaff44ad09",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0078_104948_DSC01505.jpg"
  },
  {
    "_id": "6564d86ac2d5a4eaff44ad0b",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0015_104657_DSC01443b.jpg"
  },
  {
    "_id": "6564d86bc2d5a4eaff44ad0d",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0002_104658_DSC01525.jpg"
  },
  {
    "_id": "6564d86bc2d5a4eaff44ad0f",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104659_DSC09137-Edit.jpg"
  },
  {
    "_id": "6564d86bc2d5a4eaff44ad11",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0023_104914_DSC00541-1.jpg"
  },
  {
    "_id": "6564d86bc2d5a4eaff44ad13",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/0000_104254_DSC00679-Edit.jpg"
  },
  {
    "_id": "6564d86bc2d5a4eaff44ad15",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/10/104252_DSC09133-Editb.jpg"
  },
  {
    "_id": "6564d86bc2d5a4eaff44ad17",
    "images": "https://www.ufonepal.com/ufo/wp-content/uploads/2023/09/P2_104200_3198_DSC06529_ed.jpg"
  }
]
products = {}
index = 0
for product in products_list:
    if index == 11 :
        continue
    products[ product["_id"]] = product['images']
    index = index + 1


productIds = list(products.keys())
for index, url in enumerate( list(products.values())):
    print( index )
    filename = 'images/' + url.split('/')[-1]
    print( filename )
    image_type = url.split('.')[-1]
    print( image_type )
    # image = requests.get(url).content
    # open(filename, 'wb').write(image )
    read_image = read_images( filename )
    embd = image_preprocessing(read_image, 800, 800 )
    embd = compute_embedding( embd )
    products[ productIds[index]] = embd.tolist()[0]
    # print( products[productIds[index]])

    # print( products )
f = open('products.json', 'w')
f.write( json.dumps(products ))

