def bnb_fzcp(subs, max_steps, processor):
    """
    Main Branch-and-Bound driver

    Args:
        subs: the list of subproblems
        max_steps: mximal number of steps to perform
        processor: the processor

    Returns:
        number of actually performed steps
    """
    steps = 0
    while len(subs) > 0 and steps <= max_steps and processor.running:
        s = subs.pop()
        new_subs = processor.fzcp_process(s)
        subs.extend(new_subs)
        steps = steps + 1
    return steps
