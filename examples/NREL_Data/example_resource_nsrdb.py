from rex import Resource
with Resource('/nrel/nsrdb/current/nsrdb_2020.h5', hsds=True) as res:
    ghi = res['ghi', :, 500]
    print(res.dsets)
    print(res.attrs['ghi'])
    print(res.time_index)
    print(res.meta)
    print(ghi)
