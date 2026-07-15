def batch_chunks(items, batch_size=50):

    for i in range(0, len(items), batch_size):

        yield items[i:i + batch_size]