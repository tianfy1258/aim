from aim.utils import *

def get_test_image_file(request,fn):
    dataset_id = 5583 if fn[0] == '1' else 5584
    fn = fn[1:]
    dataset1 = Dataset.objects.get(dataset_id=dataset_id)
    zfile1 = get_zipfile(dataset1.dataset_filename)
    image1 = extract_one_image(zfile1, fn, False)
    response = HttpResponse(content_type="image/jpeg")
    image1.save(response, "JPEG")
    return response