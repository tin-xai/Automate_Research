# WebArena Additional Services Setup Guide

## OneStopShop (Shopping), Wiki, and Reddit (Forum)

This document explains how to download, load, run, and verify the
remaining WebArena services:

-   OneStopShop (Shopping) -- Port 7770
-   Wiki (Wikipedia via Kiwix) -- Port 8888
-   Reddit (Forum/Postmill) -- Port 9999

------------------------------------------------------------------------

# 1. OneStopShop (Shopping)

## Download Image

    cd ~/webarena_env/images
    wget -c http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar

## Load Image

    docker load -i shopping_final_0712.tar

Verify:

    docker images | grep shopping

## Run Container

    docker run --name shopping -d -p 7770:80 shopping_final_0712

## Set Base URL (Replace with your server IP)

    docker exec shopping /var/www/magento2/bin/magento setup:store-config:set     --base-url="http://131.204.27.97:7770"

    docker exec shopping mysql -u magentouser -pMyPassword magentodb     -e "UPDATE core_config_data SET value='http://131.204.27.97:7770/' WHERE path='web/secure/base_url';"

    docker exec shopping /var/www/magento2/bin/magento cache:flush

## Verify

    curl -I http://localhost:7770

------------------------------------------------------------------------

# 2. Wiki (Wikipedia via Kiwix)

## Download ZIM File

    cd ~/webarena_env/images
    wget -c http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim

## Run Kiwix Container

    docker run -d --name wikipedia       -v $(pwd):/data       -p 8888:80       ghcr.io/kiwix/kiwix-serve:3.3.0       wikipedia_en_all_maxi_2022-05.zim

## Verify

    curl -I http://localhost:8888

------------------------------------------------------------------------

# 3. Reddit (Forum / Postmill)

## Download Image

    cd ~/webarena_env/images
    wget -c http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar

## Load Image

    docker load -i postmill-populated-exposed-withimg.tar

Verify:

    docker images | grep postmill

## Run Container

    docker run --name forum -d -p 9999:80 postmill-populated-exposed-withimg

## Verify

    curl -I http://localhost:9999

------------------------------------------------------------------------

# Access from Laptop (Recommended)

Use SSH tunnel for each service:

    ssh -L 7770:localhost:7770         -L 8888:localhost:8888         -L 9999:localhost:9999         gpu5

Then open:

-   Shopping: http://localhost:7770
-   Wiki: http://localhost:8888
-   Reddit: http://localhost:9999

------------------------------------------------------------------------

# Container Management

Stop a service:

    docker stop shopping
    docker stop wikipedia
    docker stop forum

Start a service:

    docker start shopping
    docker start wikipedia
    docker start forum

Remove container:

    docker rm -f shopping
    docker rm -f wikipedia
    docker rm -f forum

------------------------------------------------------------------------

You now have the full WebArena multi-service environment running.
