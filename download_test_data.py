import gdown

def donwload_online_retail_data(output= "online_retail_II.csv"):

    # the online_retail_II.xlsx file
    url = "https://drive.google.com/file/d/1y715A5_jx1b3l884TX2xaSIagdCF6xnB/view"
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)

#excel_reader = pd.read_excel("online_retail_II.xlsx")
#excel_reader.to_csv (r'online_retail_II.csv', index = None, header=True)