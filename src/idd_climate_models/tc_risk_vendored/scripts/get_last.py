import cdsapi
import os

def download_missing_v_file():
    c = cdsapi.Client()
    
    # Create directory if it doesn't exist
    os.makedirs('/mnt/team/rapidresponse/pub/tropical-storms/data/era5/2016', exist_ok=True)
    
    fn = '/mnt/team/rapidresponse/pub/tropical-storms/data/era5/2016/era5_v_daily_2016.grib'
    
    if not os.path.isfile(fn):
        print(f'Downloading missing file: {fn}')
        
        req_v = {
            'product_type': 'reanalysis',
            'data_format': 'grib',
            'variable': 'v_component_of_wind',
            'pressure_level': ['250', '850'],
            'year': '2016',
            'month': ['01', '02', '03', '04', '05', '06',
                      '07', '08', '09', '10', '11', '12'],
            'day': ['01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24',
                    '25', '26', '27', '28', '29', '30', '31'],
            'grid': '2.0/2.0',
            'time': ['00:00', '12:00'],
        }
        
        try:
            c.retrieve('reanalysis-era5-pressure-levels', req_v, fn)
            print(f'Successfully downloaded: {fn}')
        except Exception as e:
            print(f'Failed to download {fn}: {e}')
    else:
        print(f'File already exists: {fn}')

if __name__ == "__main__":
    download_missing_v_file()