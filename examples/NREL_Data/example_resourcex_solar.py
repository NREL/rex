from rex import SolarX
with SolarX('/nrel/nsrdb/current/nsrdb_2020.h5', hsds=True) as res:
    df = res.get_SAM_lat_lon((39.7407, -105.1686))
    print(df)
