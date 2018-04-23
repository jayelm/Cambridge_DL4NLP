def evaluate_model(sess,
                   data_dir,
                   input_node,
                   target_node,
                   prediction,
                   loss,
                   embs,
                   out_form="cosine"):
    """
    Calculate median ranking of head words for 200 examples in the dev set.
    """
    # 200 dev examples
    epochs_gen = list(
        gen_epochs(data_dir, 1, 200, FLAGS.vocab_size, phase="dev"))
    assert len(epochs_gen) == 1
    batch0 = list(epochs_gen[0])
    assert len(batch0) == 1
    assert len(batch0[0]) == 2
    assert batch0[0][1].shape == (200, )
    gloss, head = batch0[0]
    preds, loss_ = sess.run(
        [prediction, loss], feed_dict={
            input_node: gloss,
            target_node: head
        })
    print("Dev loss: {}".format(loss_))

    # Given cosine embedding output, calculate similarities to other words
    sims = 1 - np.squeeze(dist.cdist(preds, embs, metric="cosine"))
    # Replace nans with 0s
    sims = np.nan_to_num(sims)
    # Get candidate IDs ordered by similarity, decreasing order
    candidate_ids = np.flip(sims.argsort(), 1)
    # For each list of 100K candidates, get the index of the groundtruth
    # prediction
    ranks = np.array([
        np.where(cands == label) for cands, label in zip(candidate_ids, head)
    ]).squeeze()
    median_rank = np.median(ranks)
    print("Dev median rank: {}".format(median_rank))
    return median_rank
