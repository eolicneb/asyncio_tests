import re

def recursive_match(src, trg, rec_numb=0):
    """
    :type src: Str
    :type trg: List[Str]
    :type rec_numb: Int
    """
    # print('RECURSIVE MATCH #', rec_numb)
    # need to find the largest string in target available in source
    # print(src, trg)
    sub_match = re.compile('.*'+'.*'.join(trg))
    sub_in_src = re.match(sub_match, src)
    # print(sub_in_src)
    if sub_in_src:
        # The whole of it matches in the source!
        return rec_numb
    
    # print('posible sub strings')
    trg_subs = ( (trg[:i], trg[i:]) for i in range(len(trg)-1,0,-1))
    for sub_trg, next_trg in trg_subs:
        # print(sub_trg, next_trg)
        sub_match = re.compile('.*'+'.*'.join(sub_trg))
        sub_in_src = re.match(sub_match, src)
        # print(sub_in_src)
        if sub_in_src:
            rec_match = recursive_match(src, next_trg, rec_numb+1)
            if rec_match:
                return rec_match


def shortestWay(source, target):
    #Put your code here
    res = 0

    src, trg = list(source), list(target)
    # print(source, trg)
    for l_trg in trg:
        if l_trg not in src:
            res = -1
    
    # need to find the largest string in target available in source
    if res == 0:
        res = recursive_match(source, trg, 1)
        res = res if res else -1
    
    print(res)
