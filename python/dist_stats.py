import numpy as np
import pickle

def testDistStats(s):
    s.append_array_stats(np.array([[1,2],[3,4]]))
    s.append_array_stats(np.array([[2,2],[4,5]]))
    s.end_iter()
    s.append_array_stats(np.array([[3,2],[3,5]]))
    s.append_array_stats(np.array([[7,2],[4,4]]))
    for k in s.list_stats():
        print k, s.get_aggregate_stat(k)

def testDistStatsMap(s):
    s.append_array_stats('a', np.array([[1,2],[3,4]]))
    s.append_array_stats('b', np.array([[2,2],[4,5]]))
    # s.end_iter()
    # s.append_array_stats('a', np.array([[3,2],[3,5]]))
    # s.append_array_stats('b', np.array([[7,2],[4,4]]))
    print s

class distStatsMap:
    ''' wrapper for a dict of DistStats '''

    def __init__(self):
        self.map = dict()

    def append_array_stats(self, blob, data):
        if blob not in self.map:
            self.map[blob] = DistStats()
        self.map[blob].append_array_stats(data)

    def iteritems(self):
        return self.map.iteritems()

    def __str__(self):
        s = ''
        for k, v in self.iteritems():
            s += k + '\n' + str(v) + '\n' 
        return s

    def __getitem__(self, key):
        return map[key]

    def end_iter(self):
        for s in self.map.itervalues():
            s.end_iter()

    def aggregate(self):
        agg = dict()
        for k, v in self.map.iteritems():
            stat_dict = v.get_all_aggregate_stats()
            agg[k] = stat_dict
        return agg

    def dump_aggregate_pickle(self, dumpfile):
        dump = self.aggregate()

        with open(dumpfile, 'w') as f:
            pickle.dump(dump, f)

class DistStats:
    ''' class to store stats for array data across batches and iterations '''

    # stats['stat name'] -> list[iteration][batch]

    def __init__(self):
        self.stats = dict()
        self.iter_done = False
        self.nbatches = 0

    def append(self, stat, val):
        '''append stat for this iteration'''
        if self.iter_done:
            for key in self.stats:
                self.stats[key].append([])
            self.iter_done = False

        if stat not in self.stats:
            self.stats[stat] = [[]]
        self.stats[stat][-1].append(val)


    def end_iter(self):
        '''end iteration, next append will start new iteration'''
        self.iter_done = True
        for key in self.stats:
            if self.nbatches and len(self.stats[key][-1]) != self.nbatches:
                print 'Error: end_iter called after %d batches, expecting %d batches' % (self.stats[key][-1], self.nbatches)
            self.nbatches = len(self.stats[key][-1])
        

    def get_stat(self, stat):
        return self.stats[stat]


    def list_stats(self):
        return self.stats.keys()


    def append_array_stats(self, data):
        self.append('min',data.min())
        self.append('max',data.max())
        self.append('mean',data.mean())
        self.append('std',data.std())


    def get_aggregate_stat(self, stat):
        ''' aggregate stat across batches '''

        s = self.stats[stat]
        ns = np.array(s)

        # apply appropriate operation to aggregate stats across batches
        if stat == 'min':
            agg = ns.min(axis=1)
        if stat == 'max':
            agg = ns.max(axis=1)
        if stat == 'mean':
            agg = ns.mean(axis=1)
        if stat == 'std':
            agg = ns.mean(axis=1)
        return list(agg)


    def get_all_aggregate_stats(self):
        '''get stats as a list of stat names and 2d list of stats [stat,iteration]'''
        d = dict()
        for n in self.stats:
            d[n] = self.get_aggregate_stat(n)
        return d


    def __str__(self):
        return str(self.stats) + \
            ' iter_done=' + str(self.iter_done) + \
            ' nbatches=' + str(self.nbatches) 
