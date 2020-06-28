# Shopee Competitions - Product Detection

### Useful Links
- [Google Drive - Announcements](https://drive.google.com/drive/folders/1V_sHZN2MmhcfeVao3hoJpjRWLSnm1_Pe)
- [Kaggle Competition Page](https://www.kaggle.com/c/shopee-product-detection-student/)

### Categories
- 00: dresses
- 01: also dressees, pretty similar to 00
- 02: shirts
- 03: sweaters
- 04: pants / jeans
- 05: rings
- 06: earrings
- 07: caps, hats
- 08: wallets
- 09: suitcases
- 10: smart phones (non apple)
- 11: iphones / i*
- 12: clocks
- 13: milk bottles
- 14: eletric pots
- 15: coffee beams / instant coffee
- 16: shoes
- 17: high heels / boots
- 18: air conditioners
- 19: USBs / SD Cards
- 20: chairs / sofa
- 21: tennis rackets
- 22: helmets
- 23: gloves
- 24: watches
- 25: belts
- 26: earphones
- 27: toy cars
- 28: jackets
- 29: jeans / pants
- 30: sneakers
- 31: snacks
- 32: masks
- 33: hand sanitizers
- 34: cosmetic
- 35: shampoo / perfume / bottle-like
- 36: misc / 五金
- 37: laptops / computers / 3C
- 38: dining set / 餐具
- 39: vase / 園藝容具 / 造景
- 40: shower head
- 41: pillows / sofa / curtains

> 早知道用中文寫......

### Data Splitting

```python3
python3 data-split.py
```

It will generate three files, `data/train.pkl`, `data/valid.pkl` and `data/test.pkl`. The format of the files are listed in `data-split.py`

