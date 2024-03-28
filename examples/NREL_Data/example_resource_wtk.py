from rex import WindResource
with WindResource('/nrel/wtk/conus/wtk_conus_2007.h5', hsds=True) as res:
    ws88 = res['windspeed_88m', :, 1000]
    print(res.dsets)
    print(ws88)
