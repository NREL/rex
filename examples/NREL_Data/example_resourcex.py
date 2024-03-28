from rex import ResourceX
with ResourceX('/nrel/wtk/conus/wtk_conus_2007.h5', hsds=True) as res:
    df = res.get_lat_lon_df('temperature_2m', (39.7407, -105.1686))
    print(df)
