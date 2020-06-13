from django.db import models

class Cases(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT)
    date = models.DateTimeField('Date')
    cases = models.IntegerField('Count', null=True)    

    class Meta:
        unique_together = ['date', 'region']

    def __str__(self):
        return '{} Cases as of '.format(self.region, self.date.strftime('%b %d'))

class Deaths(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT)
    date = models.DateTimeField('Date')
    deaths = models.IntegerField('Count')

    class Meta:
        unique_together = ['date', 'region']

    def __str__(self):
        return '{} Deaths as of '.format(self.region, self.date.strftime('%b %d'))

class Tests(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT)
    date = models.DateTimeField('Date')
    tests = models.FloatField('Count', null=True)

    class Meta:
        unique_together = ['date', 'region']

    def __str__(self):
        return '{} Tests as of '.format(self.region.name, self.date.strftime('%b %d'))

class Country(models.Model):
    name = models.CharField('Name', max_length=100)
    alt1 = models.CharField('Alternative Name 1', max_length=200, null=True)
    alt2 = models.CharField('Alternative Name 2', max_length=200, null=True)
    alpha2 = models.CharField('Alpha-2 Code', max_length=2, null=True)
    alpha3 = models.CharField('Alpha-3 Code', max_length=3, null=True)
    numeric = models.PositiveSmallIntegerField('Numeric Code', null=True)

    def __str__(self):
        return '{}'.format(self.alpha3)

class City(models.Model):
    name = models.CharField('Name', max_length=100)
    region = models.ForeignKey('Region', on_delete=models.PROTECT)

class Region(models.Model):
    name = models.CharField('Name', max_length=100)
    name_alt = models.CharField('Alternative Name', null=True, max_length=100)
    code =  models.CharField('Region Code', null=True, max_length=10)
    country_key = models.ForeignKey('Country', on_delete=models.PROTECT, null=True)
    country = models.CharField('Country', max_length=50)
    country_alt = models.CharField('Alternative Country Name', null=True, max_length=100)
    
    longitude = models.FloatField('Longitude', null=True)
    latitude = models.FloatField('Latitude', null=True)
    
    LAND_A_KM = models.FloatField('Land Area', null=True)
    land_density = models.FloatField('Land Density', null=True)
    city_dens =  models.FloatField('Density of Largest City', null=True)
    population = models.PositiveIntegerField('Population', null=True)

    A85PLUSB = models.PositiveIntegerField('Population 85 and Older', null=True)
    A80PLUSB = models.PositiveIntegerField('Population 80 and Older', null=True)
    A75PLUSB = models.PositiveIntegerField('Population 75 and Older', null=True)
    A70PLUSB = models.PositiveIntegerField('Population 70 and Older', null=True)
    A65PLUSB = models.PositiveIntegerField('Population 65 and Older', null=True)

    A09UNDERB = models.PositiveIntegerField('Population 9 and Under', null=True)
    A14UNDERB = models.PositiveIntegerField('Population 14 and Under', null=True)
    A19UNDERB = models.PositiveIntegerField('Population 19 and Under', null=True)
    A24UNDERB = models.PositiveIntegerField('Population 24 and Under', null=True)
    A29UNDERB = models.PositiveIntegerField('Population 29 and Under', null=True)
    A34UNDERB = models.PositiveIntegerField('Population 34 and Under', null=True)

    # School goers
    A05_14B = models.PositiveIntegerField('Population 5 to 14', null=True)
    A05_19B = models.PositiveIntegerField('Population 5 to 19', null=True) 
    A05_24B = models.PositiveIntegerField('Population 5 to 24', null=True) 
    A05_29B = models.PositiveIntegerField('Population 5 to 29', null=True) 
    A05_34B = models.PositiveIntegerField('Population 5 to 34', null=True) 

    # Young Millenials
    A15_24B = models.PositiveIntegerField('Population 15 to 24', null=True) 
    A15_29B = models.PositiveIntegerField('Population 15 to 29', null=True) 
    A15_34B = models.PositiveIntegerField('Population 15 to 34', null=True) 

    # Millenials
    A20_24B = models.PositiveIntegerField('Population 20 to 24', null=True) 
    A20_29B = models.PositiveIntegerField('Population 20 to 29', null=True) 
    A20_34B = models.PositiveIntegerField('Population 20 to 34', null=True) 
    
    # Middle Age
    A35_54B = models.PositiveIntegerField('Population 35 to 54', null=True) 
    A40_54B = models.PositiveIntegerField('Population 40 to 54', null=True) 
    A45_54B = models.PositiveIntegerField('Population 45 to 54', null=True) 

    # Middle Age +
    A35_64B = models.PositiveIntegerField('Population 35 to 64', null=True) 
    A40_64B = models.PositiveIntegerField('Population 40 to 64', null=True) 
    A45_64B = models.PositiveIntegerField('Population 45 to 64', null=True) 

    A80_84B = models.PositiveIntegerField('Population 80 to 84', null=True) 
    A75_79B = models.PositiveIntegerField('Population 75 to 79', null=True) 
    A70_74B = models.PositiveIntegerField('Population 70 to 74', null=True) 
    A65_69B = models.PositiveIntegerField('Population 65 to 69', null=True) 
    A60_64B = models.PositiveIntegerField('Population 60 to 64', null=True) 
    A55_59B = models.PositiveIntegerField('Population 55 to 59', null=True)
    A50_54B = models.PositiveIntegerField('Population 50 to 54', null=True)
    A45_49B = models.PositiveIntegerField('Population 45 to 49', null=True)
    A40_44B = models.PositiveIntegerField('Population 40 to 44', null=True)
    A35_39B = models.PositiveIntegerField('Population 35 to 39', null=True)
    A30_34B = models.PositiveIntegerField('Population 30 to 34', null=True)
    A25_29B = models.PositiveIntegerField('Population 25 to 29', null=True)
    A20_24B = models.PositiveIntegerField('Population 20 to 24', null=True)
    A15_19B = models.PositiveIntegerField('Population 15 to 19', null=True)
    A10_14B = models.PositiveIntegerField('Population 10 to 14', null=True)
    A05_09B = models.PositiveIntegerField('Population 5 to 9', null=True)
    A00_04B  = models.PositiveIntegerField('Population 0 to 4', null=True)

class Measurements(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT)
    date = models.DateTimeField('Date')
    longitude = models.FloatField('Longitude')
    latitude = models.FloatField('Latitude')

    temp = models.FloatField('Temperature', null=True)
    dewpoint = models.FloatField('Dewpoint Temperature', null=True)
    uvb = models.FloatField('UV Radiation', null=True)
    evap = models.FloatField('Evaporation', null=True)

class Pollutant(models.Model):
    date = models.DateTimeField('Date')
    city = models.ForeignKey('City', on_delete=models.PROTECT)

    pollutant = models.CharField('Pollutant', max_length=20)
    count = models.PositiveSmallIntegerField('Count')
    minimum = models.FloatField('Minimum')
    maximum = models.FloatField('Maximum')
    median = models.FloatField('Median')
    variance = models.FloatField('Variance')

    class Meta:
        unique_together = ['date', 'city', 'pollutant']

class Strindex(models.Model):
    country = models.ForeignKey('Country', on_delete=models.PROTECT)
    date = models.DateTimeField('Date')

    c1 = models.IntegerField('School Closing', null=True)
    c2 = models.IntegerField('Workplace Closing', null=True)
    c3 = models.IntegerField('Cancel Public Events', null=True)
    c4 = models.IntegerField('Restrictions on Gatherings', null=True)
    c5 = models.IntegerField('Close Public Transport', null=True)
    c6 = models.IntegerField('Stay-at-Home Requirements', null=True)
    c7 = models.IntegerField('Restrictions on Internal Movement', null=True)
    c8 = models.IntegerField('International Travel Controls', null=True)
    e1 = models.IntegerField('Income Support', null=True)
    e2 = models.IntegerField('Debt / Contract Relief', null=True)
    e3 = models.IntegerField('Fiscal Measures', null=True)
    e4 = models.IntegerField('International Support', null=True)
    h1 = models.IntegerField('Public Information Campaigns', null=True)
    h2 = models.IntegerField('Testing Policy', null=True)
    h3 = models.IntegerField('Contact Tracing', null=True)
    h4 = models.IntegerField('Emergency Investment in Health Care', null=True)
    h5 = models.IntegerField('Investment in Vaccines', null=True)

    strindex = models.FloatField('Stringency Index', null=True)

    class Meta:
        unique_together = ['date', 'country']

class Mobility(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT)
    date = models.DateTimeField('Date')
    retail_n_rec = models.FloatField('Retail & Recreation')
    groc_n_pharm = models.FloatField('Grocery & Pharmacy')
    parks = models.FloatField('Parks')
    transit = models.FloatField('Transit')
    workplaces = models.FloatField('Work Places')
    residential = models.FloatField('Residential')

    # class Meta:
    #     unique_together = ['date', 'region']

class Cause(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT, null=True)
    country = models.ForeignKey('Country', on_delete=models.PROTECT, null=True)
    year = models.IntegerField('Year')

    infectious = models.IntegerField('Infectious', null=True)
    neoplasms = models.IntegerField('NeoPlasms', null=True)
    blood = models.IntegerField('Blood-based', null=True)
    endo = models.IntegerField('Endocrine', null=True)
    mental = models.IntegerField('Mental', null=True)
    nervous = models.IntegerField('Nervous System', null=True)
    circul = models.IntegerField('Circulatory', null=True)
    infectious = models.IntegerField('Infectious', null=True)
    respir = models.IntegerField('Respiratory', null=True)
    digest = models.IntegerField('Digestive', null=True)
    skin = models.IntegerField('Skin-related', null=True)
    musculo = models.IntegerField('Musculo-skeletal', null=True)
    genito = models.IntegerField('Genitourinary', null=True)
    childbirth = models.IntegerField('Maternal and Childbirth', null=True)
    perinatal = models.IntegerField('Perinatal', null=True)
    congenital = models.IntegerField('Congenital', null=True)
    other = models.IntegerField('Other', null=True)
    external = models.IntegerField('External', null=True)

class Travel(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT)
    year = models.IntegerField('Year')

    visitors = models.FloatField('Visitors')

class GDP(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT)

    year = models.IntegerField('Year')
    gdp = models.FloatField('GDP PPP, 2016, Current Dollars', null=True)

class AppleMobility(models.Model):
    region = models.ForeignKey('Region', on_delete=models.PROTECT)
    date = models.DateTimeField('Date')

    driving = models.FloatField('Driving', null=True)
    walking = models.FloatField('Walking', null=True)
    transit = models.FloatField('Transit', null=True)
    
    class Meta:
        unique_together = ['date', 'region']
