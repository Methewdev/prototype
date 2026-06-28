"""
=====================================================
GOOGLE PLAY SCRAPER
=====================================================
"""

import pandas as pd

from google_play_scraper import reviews
from google_play_scraper import Sort


APP_MAPPING = {

    "Livin' by Mandiri":
        "id.bmri.livin",

    "BRImo":
        "id.co.bri.brimo",

    "myBCA":
        "com.bca.mybca.omni.android",

    "SeaBank":
        "com.seabank.seabank",

    "Jenius":
        "com.btpn.dc"

}


def scrape_google_play(
        app_name,
        total_review
):

    app_id = APP_MAPPING[app_name]

    result, _ = reviews(

        app_id,

        lang="id",

        country="id",

        sort=Sort.NEWEST,

        count=total_review

    )

    data = []

    for r in result:

        data.append({

            "userName": r["userName"],

            "score": r["score"],

            "content": r["content"],

            "tanggal": r["at"].strftime("%Y-%m-%d")

        })

    return pd.DataFrame(data)
