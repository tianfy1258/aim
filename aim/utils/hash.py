def md5(data):
    import hashlib
    md5 = hashlib.md5()
    md5.update(data.encode())
    return md5.hexdigest()
