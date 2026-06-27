"""
=====================================================
GOOGLE PLAY SCRAPER
=====================================================
Mengambil Review Google Play Store
=====================================================
"""

import pandas as pd

from google_play_scraper import reviews, Sort


def scrape_google_play(
    app_id,
    count=500
):

    result, _ = reviews(

        app_id,

        lang="id",

        country="id",

        sort=Sort.NEWEST,

        count=count

    )

    data = []

    for review in result:

        data.append({

            "userName": review["userName"],

            "score": review["score"],

            "content": review["content"],

            "tanggal": review["at"].strftime("%Y-%m-%d")

        })

    df = pd.DataFrame(data)

    return df
