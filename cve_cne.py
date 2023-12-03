import time
import logging as lg
import re
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString # pip install beautifulsoup4
# the env also needs to install lxml;
import nvdlib # pip install nvdlib

headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
    # "Cookie": "" 
}
def get_nvd_info(cve):
    page = 'https://nvd.nist.gov/vuln/detail/'+cve
    links = []
    cwe = ()
    res = requests.get(url=page,  headers=headers)
    while (res.status_code != 200):
        time.sleep(5) # Prevent frequent visits
        res = requests.get(url=page,  headers=headers)
    try:
        soup = BeautifulSoup(res.text, 'lxml')
        tbody = soup.find(attrs={'data-testid': "vuln-hyperlinks-table"}).tbody
        
        for tr in tbody.children:
            if isinstance(tr, NavigableString): continue
            tds = tr.findAll('td')
            if 'Patch' in tds[1].text:
                links.append(tds[0].a['href'])
        tbody = soup.find(attrs={'data-testid': "vuln-CWEs-table"}).tbody
        for tr in tbody.children:
            if isinstance(tr, NavigableString): continue
            tds = tr.findAll('td')
            # cwe = (tds[0].text, tds[1].text)
            cwe = (clean_cwe(tds[0].text), tds[1].text)
            
        # Fetching the CVSS Score 2 TODO


        source_element = soup.find('span', {'data-testid': 'vuln-current-description-source'})
        # print(source_element)
        source = source_element.text.strip()
        # print(source)
    except Exception as e:
    # time.sleep(5) # Prevent frequent visits
        print(page, e)


    #     try:
    #         print(e, "we set it to nvdlib.searchCVE(cveId=cve)[0].score")
    #         cvss2_score = nvdlib.searchCVE(cveId=cve)[0].score
    #         cvss_severity = nvdlib.searchCVE(cveId=cve)[0].severity
    #     except Exception as e:
    #         print(e, "we set it to NVD-CVSS2-Other")
    #         cvss2_score = 'NVD-CVSS2-Other'
    #         cvss_severity = 'NVD-CVSS2-Other'
    
    try: 
        cvss = soup.find('span', {'class': 'severityDetail'}).text
        cvss = cvss.text.strip()
    except Exception as e:
        print(e, "we set it to NVD-CVSS2-Other")
        
    print(cvss)
        
    # print(cvss2_score)
    return cve, links, cwe, source, cvss #cvss2_score
def clean_cwe(cwe):
    cwe_pattern = r"CWE-(\d+)"
    match = re.search(cwe_pattern, cwe)
    if match:
        return match.group(0)
    else:
        return 'NVD-CWE-Other'
# main function
if __name__ == '__main__':
    
    # df = pd.read_csv("./patches.csv")
    
    # cve_list = df['CVE_ID'].to_list()
    # cve_info = []
    
    # ## get the cve info and save into csv
    # for cve in cve_list:
    #     cve, links, cwe, cne = get_nvd_info(cve)
    #     # save the info into csv
    #     cve_info.append([cve, links, cwe, cne])
    
    # cve_info_df = pd.DataFrame(cve_info, columns=['cve', 'links', 'cwe', 'cne'])
    # cve_info_df.to_csv("./cve_info.csv", index=False)
        
    # #################################
    # cna_df = pd.read_csv("./cna.csv")
    # # columns: CVE_ID,Patch_Commit,Miss,Alias,Request,Confirmation,Valid
    # # # columns: cve,Patch_Commit,Miss,Alias,Request,Confirmation,Valid
    
    # cve_info_df = pd.read_csv("./cve_info.csv")
    # # columns: cve,links,cwe,cne
    
    # df = cna_df.merge(cve_info_df, on='cve', how='left')
    
    # print(df.shape)
    # df.to_csv("./cna_report.csv", index=False)
    
    # cve, links, cwe, cne, cvss2_score = get_nvd_info('CVE-2015-1867')
    