"""heaps

This code extends heapq functionality for max-heaps and minmax-heaps [1].

[1] M.D. Atkinson, J.-R. Sack, N. Santoro, and T. Strothotte.
Min-Max Heaps and Generalized Priority Queues. Communications of 
the ACM, 29(10):996-1000, 1986.
"""

import heapq
from math import log, floor

def maxheappeek(heap):
    """Return the largest item on the heap"""
    return heap[0] if len(heap) > 0 else -float('Inf')

def maxheappop(heap):
    """Pop the largest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
    else:
        returnitem = lastelt
    return returnitem

def maxheappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)

def maxheapreplace(heap, item):
    """Pop and return the current largest value, and add the new item.

    This is more efficient than heappop() followed by heappush(), and can be
    more appropriate when using a fixed-size heap.  Note that the value
    returned may be smaller than item!  That constrains reasonable uses of
    this routine unless written as part of a conditional replacement:

        if item > heap[0]:
            item = heapreplace(heap, item)
    """
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    heapq._siftup_max(heap, 0)
    return returnitem


def maxheappushpop(heap, item):
    """Fast version of a heappush followed by a heappop."""
    if heap and heapq.cmp_lt(item, heap[0]):
        item, heap[0] = heap[0], item
        heapq._siftup_max(heap, 0)
    return item

def mmheappush(heap, item):
    """Push item onto heap, maintaining the minmax-heap invariant."""
    
    heap.append(item)
    _bubbleup(heap, len(heap) - 1)


def mmheappeekmin(heap):
    """Return the smallest item on the heap"""
    
    return heap[0] if len(heap) > 0 else float('Inf')

def mmheappeekmax(heap):
    """Return the largest item on the heap"""
    
    maxpos = _maxindex(heap)
    return heap[maxpos] if len(heap) > 0 else -float('Inf')

def mmheappopmin(heap,):
    """Pop the smallest item off heap, maintaining the minmax-heap invariant."""
    
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _trickledown(heap, 0)
    else:
        returnitem = lastelt
    return returnitem

def mmheappopmax(heap):
    """Pop the largest item off heap, maintaining the minmax-heap invariant."""
    
    maxpos = _maxindex(heap)
    if len(heap) > 3:
        lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
        returnitem = heap[maxpos]
        heap[maxpos] = lastelt
        _trickledown(heap, maxpos)
    elif len(heap) == 3 and maxpos == 1:
        lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
        returnitem = heap[maxpos]
        heap[maxpos] = lastelt
    else:
        returnitem = heap.pop()
    return returnitem


def mmheapminreplace(heap, item):
    """Pop and return the current smallest value, and add the new item.

    This is more efficient than heappop() followed by heappush(), and can be
    more appropriate when using a fixed-size heap.  Note that the value
    returned may be larger than item!  That constrains reasonable uses of
    this routine unless written as part of a conditional replacement:

        if item > heap[0]:
            item = heapreplace(heap, item)
    """
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    _trickledown(heap, 0)
    return returnitem

def _maxindex(heap):
    if len(heap) == 0:
        raise IndexError()
    elif 0 < len(heap) < 3:
        return len(heap) - 1
    else:
        return 2 if heapq.cmp_lt(heap[1], heap[2]) else 1

def _trickledown(heap, pos):
    if _on_min_level(pos):
        _trickledown_min(heap, pos)
    else:
        _trickledown_max(heap, pos)

def _trickledown_min(heap, pos):
    
    while 2*pos + 1 < len(heap):
        m = 2*pos + 1
        for d in [2*pos + 2, 4*pos + 3, 4*pos + 4, 4*pos + 5, 4*pos + 6]:
            if d < len(heap) and heapq.cmp_lt(heap[d], heap[m]):
                m = d
                
        if m > 2*pos + 2:

            if heapq.cmp_lt(heap[m], heap[pos]):
                heap[m], heap[pos] = heap[pos], heap[m]
                if heapq.cmp_lt(heap[(m-1) >> 1], heap[m]):
                    heap[m], heap[(m-1) >> 1] = heap[(m-1) >> 1], heap[m]
                pos = m
                continue
                
        else:
            if heapq.cmp_lt(heap[m], heap[pos]):
                heap[m], heap[pos] = heap[pos], heap[m]
        break

def _trickledown_max(heap, pos):
    
    while 2*pos + 1 < len(heap):
        m = 2*pos + 1
        for d in [2*pos + 2, 4*pos + 3, 4*pos + 4, 4*pos + 5, 4*pos + 6]:
            if d < len(heap) and heapq.cmp_lt(heap[m], heap[d]):
                m = d

        if m > 2*pos + 2:
            
            if heapq.cmp_lt(heap[pos], heap[m]):
                heap[m], heap[pos] = heap[pos], heap[m]
                if heapq.cmp_lt(heap[m], heap[(m-1)>>1]):
                    heap[m], heap[(m-1) >> 1] = heap[(m-1) >> 1], heap[m]
                pos = m
                continue
                
        else:
            if heapq.cmp_lt(heap[pos], heap[m]):
                heap[m], heap[pos] = heap[pos], heap[m]
        break

def _bubbleup(heap, pos):
    parentpos = (pos - 1) >> 1
    if _on_min_level(pos):
        if parentpos >= 0 and heapq.cmp_lt(heap[parentpos], heap[pos]):
            heap[pos], heap[parentpos] = heap[parentpos], heap[pos]
            _bubbleup_max(heap, parentpos)
        else:
            _bubbleup_min(heap, pos)
    else:
        if parentpos >= 0 and heapq.cmp_lt(heap[pos], heap[parentpos]):
            heap[pos], heap[parentpos] = heap[parentpos], heap[pos]
            _bubbleup_min(heap, parentpos)
        else:
            _bubbleup_max(heap, pos)
            
def _bubbleup_min(heap, pos):
    grandparentpos = (((pos - 1) >> 1) - 1) >> 1
    while grandparentpos > 0:
        if heapq.cmp_lt(heap[pos], heap[grandparentpos]):
            heap[pos], heap[grandparentpos] = heap[grandparentpos], heap[pos]
            pos = grandparentpos
            grandparentpos = (((pos - 1) >> 1) - 1) >> 1
            continue
        break

def _bubbleup_max(heap, pos):
    grandparentpos = (((pos - 1) >> 1) - 1) >> 1
    while grandparentpos > 0:
        if heapq.cmp_lt(heap[grandparentpos], heap[pos]):
            heap[pos], heap[grandparentpos] = heap[grandparentpos], heap[pos]
            pos = grandparentpos
            grandparentpos = (((pos - 1) >> 1) - 1) >> 1
            continue
        break

def _on_min_level(i):
    return (_level(i) % 2) == 0

def _level(i):
    return int(floor(log(i+1, 2)))
    
    
def main():
    
    heap = []
    mmheappush(heap, 1)
    mmheappush(heap, 2)
    mmheappush(heap, 10)
    mmheappush(heap, 5)
    mmheappush(heap, 9)
    print heap
    
    heaps = [-float('Inf')]*5
    mmheapminreplace(heap, 1)
    mmheapminreplace(heap, 2)
    mmheapminreplace(heap, 10)
    mmheapminreplace(heap, 5)
    mmheapminreplace(heap, 9)
    print heap

if __name__ == '__main__':
    main()
    