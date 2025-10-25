#!/usr/bin/env python
# coding: utf-8

# # Анализ лояльности пользователей Яндекс Афиши

# ## Этапы выполнения проекта
# 
# ### 1. Загрузка данных и их предобработка
# 
# ---
# 
# **Задача 1.1:** Напишите SQL-запрос, выгружающий в датафрейм pandas необходимые данные. Используйте следующие параметры для подключения к базе данных `data-analyst-afisha`:
# 
# - **Хост** — `rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net`
# - **База данных** — `data-analyst-afisha`
# - **Порт** — `6432`
# - **Аутентификация** — `Database Native`
# - **Пользователь** — `praktikum_student`
# - **Пароль** — `Sdf4$2;d-d30pp`
# 
# Для выгрузки используйте запрос из предыдущего урока и библиотеку SQLAlchemy.
# 
# Выгрузка из базы данных SQL должна позволить собрать следующие данные:
# 
# - `user_id` — уникальный идентификатор пользователя, совершившего заказ;
# - `device_type_canonical` — тип устройства, с которого был оформлен заказ (`mobile` — мобильные устройства, `desktop` — стационарные);
# - `order_id` — уникальный идентификатор заказа;
# - `order_dt` — дата создания заказа (используйте данные `created_dt_msk`);
# - `order_ts` — дата и время создания заказа (используйте данные `created_ts_msk`);
# - `currency_code` — валюта оплаты;
# - `revenue` — выручка от заказа;
# - `tickets_count` — количество купленных билетов;
# - `days_since_prev` — количество дней от предыдущей покупки пользователя, для пользователей с одной покупкой — значение пропущено;
# - `event_id` — уникальный идентификатор мероприятия;
# - `service_name` — название билетного оператора;
# - `event_type_main` — основной тип мероприятия (театральная постановка, концерт и так далее);
# - `region_name` — название региона, в котором прошло мероприятие;
# - `city_name` — название города, в котором прошло мероприятие.
# 
# ---
# 

# In[1]:


# Используйте ячейки типа Code для вашего кода,
# а ячейки типа Markdown для комментариев и выводов


# In[2]:


# При необходимости добавляйте новые ячейки для кода или текста


# In[3]:


#!pip install sqlalchemy
#№!pip install psycopg2
#!pip install psycopg2-binary
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
np.set_printoptions(threshold=np.inf) 
import matplotlib.pyplot as plt
import matplotlib as mb
import seaborn as sns
get_ipython().system('pip install scypy')
get_ipython().system('pip install phik')
from phik import phik_matrix


# In[4]:


db_config = {'user': 'praktikum_student', # имя пользователя
             'pwd': 'Sdf4$2;d-d30pp', # пароль
             'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
             'port': 6432, # порт подключения
             'db': 'data-analyst-afisha' # название базы данных
             }


# In[5]:


connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
    db_config['user'],
    db_config['pwd'],
    db_config['host'],
    db_config['port'],
    db_config['db'],
)


# In[6]:


engine = create_engine(connection_string)


# In[7]:


query = '''
select
  user_id,
  device_type_canonical,
  order_id,
  created_dt_msk as order_dt,
  created_ts_msk as order_ts,
  currency_code,
  revenue,
  tickets_count,
  extract('days' from created_dt_msk - lag(created_dt_msk) over(partition by user_id order by created_dt_msk)) as days_since_prev,
  event_id,
  event_name_code as event_name,
  event_type_main,
  service_name,
  region_name,
  city_name
from afisha.purchases
left join afisha.events using (event_id)
left join afisha.city using (city_id)
left join afisha.regions using (region_id)
where device_type_canonical in ('mobile', 'desktop') and event_type_main != 'фильм'
order by user_id
'''


# In[8]:


df = pd.read_sql_query(query, con=engine)


# ---
# 
# **Задача 1.2:** Изучите общую информацию о выгруженных данных. Оцените корректность выгрузки и объём полученных данных.
# 
# Предположите, какие шаги необходимо сделать на стадии предобработки данных — например, скорректировать типы данных.
# 
# Зафиксируйте основную информацию о данных в кратком промежуточном выводе.
# 
# ---

# In[9]:


df.info()


# На стадии предобработки данных можно скорректировать типы данных (object) и понизить их разрядность (если это возможно). Возможно заменить пустые данные значениями. Удалить дубликаты если они есть.

# ---
# 
# ###  2. Предобработка данных
# 
# Выполните все стандартные действия по предобработке данных:
# 
# ---
# 
# **Задача 2.1:** Данные о выручке сервиса представлены в российских рублях и казахстанских тенге. Приведите выручку к единой валюте — российскому рублю.
# 
# Для этого используйте датасет с информацией о курсе казахстанского тенге по отношению к российскому рублю за 2024 год — `final_tickets_tenge_df.csv`. Его можно загрузить по пути `https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv')`
# 
# Значения в рублях представлено для 100 тенге.
# 
# Результаты преобразования сохраните в новый столбец `revenue_rub`.
# 
# ---
# 

# In[10]:


exchange = pd.read_csv('https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv')


# In[11]:


# преобразовываем формат из object в формат даты, иначе выдает ошибку
exchange['data'] = exchange['data'].astype('datetime64[ns]')


# In[12]:


merged_df = pd.merge(df, exchange, left_on=['order_dt', 'currency_code'],right_on=['data','cdx'], how='left')


# In[13]:


merged_df['revenue_rub'] = merged_df.apply(lambda row: row['revenue'] * (row['curs'] / row['nominal'])                                            if row['currency_code'] == 'kzt' else row['revenue'], axis=1)


# In[14]:


# удаляю лишние столбцы 
merged_df = merged_df.drop(['data','nominal','curs','cdx'], axis=1)


# ---
# 
# **Задача 2.2:**
# 
# - Проверьте данные на пропущенные значения. Если выгрузка из SQL была успешной, то пропуски должны быть только в столбце `days_since_prev`.
# - Преобразуйте типы данных в некоторых столбцах, если это необходимо. Обратите внимание на данные с датой и временем, а также на числовые данные, размерность которых можно сократить.
# - Изучите значения в ключевых столбцах. Обработайте ошибки, если обнаружите их.
#     - Проверьте, какие категории указаны в столбцах с номинальными данными. Есть ли среди категорий такие, что обозначают пропуски в данных или отсутствие информации? Проведите нормализацию данных, если это необходимо.
#     - Проверьте распределение численных данных и наличие в них выбросов. Для этого используйте статистические показатели, гистограммы распределения значений или диаграммы размаха.
#         
#         Важные показатели в рамках поставленной задачи — это выручка с заказа (`revenue_rub`) и количество билетов в заказе (`tickets_count`), поэтому в первую очередь проверьте данные в этих столбцах.
#         
#         Если обнаружите выбросы в поле `revenue_rub`, то отфильтруйте значения по 99 перцентилю.
# 
# После предобработки проверьте, были ли отфильтрованы данные. Если были, то оцените, в каком объёме. Сформулируйте промежуточный вывод, зафиксировав основные действия и описания новых столбцов.
# 
# ---

# In[15]:


# преобразовываем типы данных
for column in ['user_id','device_type_canonical','currency_code','event_name','event_type_main','service_name'               ,'region_name','city_name']:
    merged_df[column] = merged_df[column].astype('string')
    
for column in ['revenue','days_since_prev','revenue_rub']:
    merged_df[column] = pd.to_numeric(merged_df[column], downcast='float')    


# В поле event_type_main есть значения - "другое" - которые обозначают отсутствие информации

# In[16]:


plt.figure(figsize=(10, 5))
sns.histplot(data=merged_df, x='tickets_count', bins = 100).set( title = 'Распределение значений по кол-ву билетов', 
                                                                 ylabel = 'Частота',
                                                                 xlabel = 'Кол-во билетов')
plt.show()

plt.figure(figsize=(20, 5))
sns.boxplot(data=merged_df, x = 'revenue_rub').set(title='Распределение значений выручки в руб.')


plt.show()


# In[17]:


merged_df.describe()


# In[18]:


revenue_99th_percentile = merged_df['revenue_rub'].quantile(0.99)

filtered_df = merged_df[merged_df['revenue_rub'] <= revenue_99th_percentile]


# In[19]:


# смотрим сколько данных отфильтровано
a, b = len(merged_df), len(filtered_df)
print(a, b, round((a-b)/a*100, 2))


# In[181]:


filtered_df['user_id'].nunique()


# ---
# 
# ### 3. Создание профиля пользователя
# 
# В будущем отдел маркетинга планирует создать модель для прогнозирования возврата пользователей. Поэтому сейчас они просят вас построить агрегированные признаки, описывающие поведение и профиль каждого пользователя.
# 
# ---
# 
# **Задача 3.1.** Постройте профиль пользователя — для каждого пользователя найдите:
# 
# - дату первого и последнего заказа;
# - устройство, с которого был сделан первый заказ;
# - регион, в котором был сделан первый заказ;
# - билетного партнёра, к которому обращались при первом заказе;
# - жанр первого посещённого мероприятия (используйте поле `event_type_main`);
# - общее количество заказов;
# - средняя выручка с одного заказа в рублях;
# - среднее количество билетов в заказе;
# - среднее время между заказами.
# 
# После этого добавьте два бинарных признака:
# 
# - `is_two` — совершил ли пользователь 2 и более заказа;
# - `is_five` — совершил ли пользователь 5 и более заказов.
# 
# **Рекомендация:** перед тем как строить профиль, отсортируйте данные по времени совершения заказа.
# 
# ---
# 

# In[20]:


sorted_df = filtered_df.sort_values(by=['order_ts','user_id'], ascending = True)


# In[21]:


#считаем минимальную дату заказа
min_dt = sorted_df.groupby('user_id')['order_ts'].min()


# In[22]:


#считаем максимальную дату заказа
max_dt = sorted_df.groupby('user_id')['order_ts'].max()


# In[23]:


# присоединяю столбец с минимальной датой заказа к датафрейму
df_merged = pd.merge(sorted_df, min_dt, left_on=['user_id'], right_on =['user_id'], how = 'left')  


# In[24]:


# присоединяю столбец с максимальной датой заказа к датафрейму
df_merged = pd.merge(df_merged, max_dt, left_on=['user_id'], right_on =['user_id'], how = 'left')  


# In[25]:


# переименовываю столбцы
df_merged = df_merged.rename(columns={'order_ts_x': 'order_ts', 'order_ts_y': 'min_order_ts', 'order_ts': 'max_order_ts'})


# In[26]:


#для меня - проверка, что столбцы корректные
#df_merged_1 = df_merged[(df_merged['user_id'] == '76dfb606b98b55d') & (df_merged['max_order_dt'] < '2024-10-31')]                                        


# In[27]:


# добавляю флаг первого заказа на пользователя 
df_merged['is_first_order'] = df_merged['order_ts'] == df_merged['min_order_ts']


# In[28]:


# устройство, с которого был сделан первый заказ
first_order_device = df_merged[df_merged['is_first_order'] == True].groupby('user_id')['device_type_canonical'].min()

# присоединяю столбец с устройством к датафрейму
df_merged = pd.merge(df_merged, first_order_device, left_on=['user_id'], right_on =['user_id'], how = 'left')  

# переименовываю столбец
df_merged = df_merged.rename(columns={'device_type_canonical_y': 'first_order_device', 'device_type_canonical_x': 'device_type_canonical'})


# In[29]:


#для меня - проверка, что столбцы корректные
#df_merged_1 = df_merged[(df_merged['user_id'] == 'e73089d7d016cd8')]  
#df_merged_1.sort_values(by=['order_ts','user_id'], ascending = True)


# In[30]:


#регион, в котором был сделан первый заказ
first_order_region = df_merged[df_merged['is_first_order'] == True].groupby('user_id')['region_name'].min()

# присоединяю столбец с регионом к датафрейму
df_merged = pd.merge(df_merged, first_order_region, left_on=['user_id'], right_on =['user_id'], how = 'left')  

# переименовываю столбец
df_merged = df_merged.rename(columns={'region_name_y': 'first_order_region', 'region_name_x': 'region_name'})


# In[31]:


#билетного партнёра, к которому обращались при первом заказе
first_service_name = df_merged[df_merged['is_first_order'] == True].groupby('user_id')['service_name'].min()

# присоединяю столбец с регионом к датафрейму
df_merged = pd.merge(df_merged, first_service_name, left_on=['user_id'], right_on =['user_id'], how = 'left')  

# переименовываю столбец
df_merged = df_merged.rename(columns={'service_name_y': 'first_service_name', 'service_name_x': 'service_name'})


# In[32]:


#жанр первого посещённого мероприятия (используйте поле event_type_main)
first_event_type = df_merged[df_merged['is_first_order'] == True].groupby('user_id')['event_type_main'].min()

# присоединяю столбец с регионом к датафрейму
df_merged = pd.merge(df_merged, first_event_type, left_on=['user_id'], right_on =['user_id'], how = 'left')  

# переименовываю столбец
df_merged = df_merged.rename(columns={'event_type_main_y': 'first_event_type', 'event_type_main_x': 'event_type_main'})


# In[33]:


#общее количество заказов
total_orders_cnt = sorted_df.groupby('user_id')['order_id'].count()

# присоединяю столбец с кол-вом заказов к датафрейму
df_merged = pd.merge(df_merged, total_orders_cnt, left_on=['user_id'], right_on =['user_id'], how = 'left')  

# переименовываю столбец
df_merged = df_merged.rename(columns={'order_id_y': 'total_orders_cnt', 'order_id_x': 'order_id'})


# In[34]:


#выручка на клиента в рублях
avg_revenue_rub_pre = sorted_df.groupby('user_id')['revenue_rub'].sum()

# присоединяю столбец с выручкой заказов к датафрейму
df_merged = pd.merge(df_merged, avg_revenue_rub_pre, left_on=['user_id'], right_on =['user_id'], how = 'left')  

# переименовываю столбец
df_merged = df_merged.rename(columns={'revenue_rub_y': 'total_revenue_rub', 'revenue_rub_x': 'revenue_rub'})

#средняя выручка с одного заказа в рублях
df_merged['avg_revenue_rub'] = df_merged['total_revenue_rub'] / df_merged['total_orders_cnt'] 

# удаляю лишние столбцы 
df_merged = df_merged.drop('total_revenue_rub', axis = 1)


# In[35]:


#среднее количество билетов в заказе
avg_cnt_per_order = sorted_df.groupby('user_id')['tickets_count'].mean()

# присоединяю столбец с кол-вом билетов к датафрейму
df_merged = pd.merge(df_merged, avg_cnt_per_order, left_on=['user_id'], right_on =['user_id'], how = 'left')  

# переименовываю столбец
df_merged = df_merged.rename(columns={'tickets_count_y': 'avg_cnt_per_order', 'tickets_count_x': 'tickets_count'})


# In[36]:


#среднее время между заказами
df_merged['order_ts_next'] = (df_merged.sort_values(by=['order_ts'], ascending=True).groupby(['user_id'])['order_ts'].shift(-1))


# In[37]:


#df_merged_1 = df_merged[(df_merged['user_id'] == '57ef0a1905ac488')]


# In[38]:


# создаю датафрейм для подсчета среднего времени заказа
df_merged_short = df_merged[['user_id','order_ts','order_ts_next']].dropna()


# In[39]:


df_merged_short['date_difference_d'] = (df_merged_short['order_ts_next'] - df_merged_short['order_ts']).dt.days


# In[40]:


df_avg_time = df_merged_short.groupby('user_id')['date_difference_d'].mean()


# In[41]:


df_merged = pd.merge(df_merged, df_avg_time, left_on=['user_id'], right_on =['user_id'], how = 'left')  


# In[42]:


# совершил ли пользователь 2 и более заказа;
df_merged['is_two'] = df_merged['total_orders_cnt'] >= 2

# совершил ли пользователь 5 и более заказов.
df_merged['is_five'] = df_merged['total_orders_cnt'] >= 5


# In[43]:


client_profile = df_merged[['user_id','min_order_ts','max_order_ts','first_order_device','first_order_region','first_service_name','first_event_type'                            ,'total_orders_cnt','avg_revenue_rub','avg_cnt_per_order','date_difference_d','is_two','is_five']].drop_duplicates()


# ---
# 
# **Задача 3.2.** Прежде чем проводить исследовательский анализ данных и делать выводы, важно понять, с какими данными вы работаете: насколько они репрезентативны и нет ли в них аномалий.
# 
# Используя данные о профилях пользователей, рассчитайте:
# 
# - общее число пользователей в выборке;
# - среднюю выручку с одного заказа;
# - долю пользователей, совершивших 2 и более заказа;
# - долю пользователей, совершивших 5 и более заказов.
# 
# Также изучите статистические показатели:
# 
# - по общему числу заказов;
# - по среднему числу билетов в заказе;
# - по среднему количеству дней между покупками.
# 
# По результатам оцените данные: достаточно ли их по объёму, есть ли аномальные значения в данных о количестве заказов и среднем количестве билетов?
# 
# Если вы найдёте аномальные значения, опишите их и примите обоснованное решение о том, как с ними поступить:
# 
# - Оставить и учитывать их при анализе?
# - Отфильтровать данные по какому-то значению, например, по 95-му или 99-му перцентилю?
# 
# Если вы проведёте фильтрацию, то вычислите объём отфильтрованных данных и выведите статистические показатели по обновлённому датасету.

# In[44]:


total_users_cnt = client_profile['user_id'].count()
avg_revenue_per_order = round(client_profile['avg_revenue_rub'].mean(),1)
share_2_plus_orders =  round(client_profile['is_two'].sum() / client_profile['user_id'].count(),1)*100
share_5_plus_orders =  round(client_profile['is_five'].sum() / client_profile['user_id'].count(),1)*100

print(f'Общее количество пользователей: {total_users_cnt}\nСредняя выручка с одного заказа: {avg_revenue_per_order}\nДоля пользователей, совершивших 2 и более заказа: {share_2_plus_orders}%\nДоля пользователей, совершивших 5 и более заказов: {share_5_plus_orders}%')


# In[45]:


client_profile[['total_orders_cnt','avg_cnt_per_order','date_difference_d']].describe()


# Довольно сильно различие заметно в кол-ве заказов - разюрос виден исходя из среднего и стандартного отклонения. Исключу выбросы по 99% перцентилю, так как эти данные могут исказить результаты анализа (не по 95, так как в этом случае будет исключено 5% данных, что, как мне кажется, слишком много).

# In[46]:


orders_99th_percentile = client_profile['total_orders_cnt'].quantile(0.99)

client_profile_f = client_profile[client_profile['total_orders_cnt'] <= orders_99th_percentile]


# In[47]:


# смотрим сколько данных отфильтровано
a, b = len(client_profile), len(client_profile_f)
print(f'Изначальный датафрейм: {a}\nОтфильтрованный датафрейм: {b}\nДоля: {round((a-b)/a*100, 2)}%')


# ---
# 
# ### 4. Исследовательский анализ данных
# 
# Следующий этап — исследование признаков, влияющих на возврат пользователей, то есть на совершение повторного заказа. Для этого используйте профили пользователей.

# 
# 
# #### 4.1. Исследование признаков первого заказа и их связи с возвращением на платформу
# 
# Исследуйте признаки, описывающие первый заказ пользователя, и выясните, влияют ли они на вероятность возвращения пользователя.
# 
# ---
# 
# **Задача 4.1.1.** Изучите распределение пользователей по признакам.
# 
# - Сгруппируйте пользователей:
#     - по типу их первого мероприятия;
#     - по типу устройства, с которого совершена первая покупка;
#     - по региону проведения мероприятия из первого заказа;
#     - по билетному оператору, продавшему билеты на первый заказ.
# - Подсчитайте общее количество пользователей в каждом сегменте и их долю в разрезе каждого признака. Сегмент — это группа пользователей, объединённых определённым признаком, то есть объединённые принадлежностью к категории. Например, все клиенты, сделавшие первый заказ с мобильного телефона, — это сегмент.
# - Ответьте на вопрос: равномерно ли распределены пользователи по сегментам или есть выраженные «точки входа» — сегменты с наибольшим числом пользователей?
# 
# ---
# 

# In[48]:


# считаю кол-во пользователей по типу первого мероприятия
first_event_type_g = client_profile_f.groupby('first_event_type')['user_id'].count().reset_index().sort_values(by='user_id', ascending = False)
# кол-во пользователей
total_cnt = client_profile_f['user_id'].count()
# доля клиентов по типам мероприятий
share_types = first_event_type_g['user_id']/total_cnt*100
# объединение таблиц
first_event_type_g = pd.merge(first_event_type_g, share_types,left_index=True, right_index=True, how='left')
# переименовываю столбец
first_event_type_g = first_event_type_g.rename(columns={'user_id_x': 'users_cnt', 'user_id_y': 'share'})
#вывожу результат
first_event_type_g


# По результатам группикровки, видно что для большинства клиентов концерты являются мероприятием входа на платформу - 44%.

# In[49]:


# считаю кол-во пользователей по типу первого мероприятия
first_device_g = client_profile_f.groupby('first_order_device')['user_id'].count().reset_index().sort_values(by='user_id', ascending = False)
# доля клиентов по типу устройства
share_types = first_device_g['user_id']/total_cnt*100
# объединение таблиц
first_device_g = pd.merge(first_device_g, share_types, left_index=True, right_index=True, how='left')
# переименовываю столбец
first_device_g = first_device_g.rename(columns={'user_id_x': 'users_cnt', 'user_id_y': 'share'})
#вывожу результат
first_device_g


# Большая часть клиентов начала пользоваться сервисом с мобильного устройства - 82.8%

# In[50]:


# считаю кол-во пользователей по типу первого мероприятия
first_region_g = client_profile_f.groupby('first_order_region')['user_id'].count().reset_index().sort_values(by='user_id', ascending = False)
# доля клиентов по типу устройства
share_types = first_region_g['user_id']/total_cnt*100
# объединение таблиц
first_region_g = pd.merge(first_region_g, share_types, left_index=True, right_index=True, how='left')
# переименовываю столбец
first_region_g = first_region_g.rename(columns={'user_id_x': 'users_cnt', 'user_id_y': 'share'})
#вывожу результат
first_region_g.head(10)


# У бОльшего кол-ва пользователь регион проведения мероприятия по первому заказу - Каменевский регион, 32.7%

# In[51]:


# считаю кол-во пользователей по типу первого мероприятия
first_service_g = client_profile_f.groupby('first_service_name')['user_id'].count().reset_index().sort_values(by='user_id', ascending = False)
# доля клиентов по типу устройства
share_types = first_service_g['user_id']/total_cnt*100
# объединение таблиц
first_service_g = pd.merge(first_service_g, share_types, left_index=True, right_index=True, how='left')
# переименовываю столбец
first_service_g = first_service_g.rename(columns={'user_id_x': 'users_cnt', 'user_id_y': 'share'})
#вывожу результат
first_service_g.head(10)


# "Билеты без проблем" - наиболее частый сервис для овершения первого заказа

# ---
# 
# **Задача 4.1.2.** Проанализируйте возвраты пользователей:
# 
# - Для каждого сегмента вычислите долю пользователей, совершивших два и более заказа.
# - Визуализируйте результат подходящим графиком. Если сегментов слишком много, то поместите на график только 10 сегментов с наибольшим количеством пользователей. Такое возможно с сегментами по региону и по билетному оператору.
# - Ответьте на вопросы:
#     - Какие сегменты пользователей чаще возвращаются на Яндекс Афишу?
#     - Наблюдаются ли успешные «точки входа» — такие сегменты, в которых пользователи чаще совершают повторный заказ, чем в среднем по выборке?
# 
# При интерпретации результатов учитывайте размер сегментов: если в сегменте мало пользователей (например, десятки), то доли могут быть нестабильными и недостоверными, то есть показывать широкую вариацию значений.
# 
# ---
# 

# In[52]:


# считаю кол-во пользователей по типу первого мероприятия
first_event_type_g_2 = client_profile_f.groupby(['first_event_type','is_two'])['user_id'].count().reset_index().sort_values(by=['first_event_type','is_two'], ascending = False)
# кол-во пользователей 
first_event_type_g_2 = first_event_type_g_2.loc[first_event_type_g_2['is_two'] == True,['first_event_type','user_id']]
# объединение таблиц
first_event_type_g_2 = pd.merge(first_event_type_g_2, first_event_type_g, left_on = 'first_event_type', right_on = 'first_event_type', how='left')
# удаляю лишний столбец
first_event_type_g_2 = first_event_type_g_2.drop(['share'], axis=1)
first_event_type_g_2['is_two_share'] = round(first_event_type_g_2['user_id']/first_event_type_g_2['users_cnt']*100,2)
first_event_type_g_2


# In[169]:


## график
first_event_type_graph = first_event_type_g_2.reset_index(drop=True, inplace=False)
#first_event_type_g_2.drop(columns=first_event_type_g_2.columns[index], axis=1)
first_event_type_graph =  first_event_type_graph.set_index('first_event_type', inplace=False)

first_event_type_graph[['is_two_share']].sort_values(by='is_two_share').plot(kind='bar',
    title='Доля пользователей с 2 и более заказами по типу мероприятия',
    rot = 50,
    xlabel = 'Тип первого мероприятия',
    ylabel = 'Доля пользователей с 2 и более заказами',
    legend = False)
plt.show()


# In[54]:


# считаю кол-во пользователей по типу первого мероприятия
first_device_g_2 = client_profile_f.groupby(['first_order_device','is_two'])['user_id'].count().reset_index().sort_values(by=['first_order_device','is_two'], ascending = False)
# кол-во пользователей 
first_device_g_2 = first_device_g_2.loc[first_device_g_2['is_two'] == True,['first_order_device','user_id']]
# объединение таблиц
first_device_g_2 = pd.merge(first_device_g_2, first_device_g, left_on = 'first_order_device', right_on = 'first_order_device', how='left')
# удаляю лишний столбец
first_device_g_2 = first_device_g_2.drop(['share'], axis=1)
first_device_g_2['is_two_share'] = round(first_device_g_2['user_id']/first_device_g_2['users_cnt']*100,2)
first_device_g_2


# In[167]:


## график
first_device_g_graph = first_device_g_2.reset_index(drop=True, inplace=False)
#first_event_type_g_2.drop(columns=first_event_type_g_2.columns[index], axis=1)
first_device_g_graph =  first_device_g_graph.set_index('first_order_device', inplace=False)

first_device_g_graph[['is_two_share']].sort_values(by='is_two_share').plot(kind='bar',
    title='Доля пользователей с 2 и более заказами по устройствам',
    rot = 50,
    xlabel = 'Тип устройства',
    ylabel = 'Доля пользователей с 2 и более заказами',
    legend = False
)
plt.show()


# In[56]:


# считаю кол-во пользователей по типу первого мероприятия
first_region_g_2 = client_profile_f.groupby(['first_order_region','is_two'])['user_id'].count().reset_index().sort_values(by=['first_order_region','is_two'], ascending = False)
# кол-во пользователей 
first_region_g_2 = first_region_g_2.loc[first_region_g_2['is_two'] == True,['first_order_region','user_id']]
# объединение таблиц
first_region_g_2 = pd.merge(first_region_g_2, first_region_g, left_on = 'first_order_region', right_on = 'first_order_region', how='left')
# удаляю лишний столбец
first_region_g_2 = first_region_g_2.drop(['share'], axis=1)
first_region_g_2['is_two_share'] = round(first_region_g_2['user_id']/first_region_g_2['users_cnt']*100,2)
first_region_g_2 = first_region_g_2.sort_values(by='users_cnt', ascending = False)
first_region_g_2.head(15)


# In[93]:


a = first_region_g_2.tail(10)
a['is_two_share'].mean()


# In[139]:


b = first_region_g_2.head(10)
#b['is_two_share'].mean()
b = b.reset_index(drop=True, inplace=False)


# In[141]:


b = b[['is_two_share','first_order_region']]


# In[143]:


b = b.set_index('first_order_region', inplace=False)


# In[165]:


## график
b.sort_values(by='is_two_share').plot(kind='barh',
    title='Доля пользователей с 2 и более заказами по региону',
    rot = 10,
    xlabel = 'Регион 1го заказа',
    ylabel = 'Доля пользователей с 2 и более заказами',
    legend = False
)
plt.show()


# In[60]:


# считаю кол-во пользователей по типу первого мероприятия
first_service_g_2 = client_profile_f.groupby(['first_service_name','is_two'])['user_id'].count().reset_index().sort_values(by=['first_service_name','is_two'], ascending = False)
# кол-во пользователей 
first_service_g_2 = first_service_g_2.loc[first_service_g_2['is_two'] == True,['first_service_name','user_id']]
# объединение таблиц
first_service_g_2 = pd.merge(first_service_g_2, first_service_g, left_on = 'first_service_name', right_on = 'first_service_name', how='left')
# удаляю лишний столбец
first_service_g_2 = first_service_g_2.drop(['share'], axis=1)
first_service_g_2['is_two_share'] = round(first_service_g_2['user_id']/first_service_g_2['users_cnt']*100,2)
first_service_g_2 = first_service_g_2.sort_values(by='is_two_share', ascending = False)
first_service_g_2


# In[149]:


## график
c = first_service_g_2.head(10)
#b['is_two_share'].mean()
c = c.reset_index(drop=True, inplace=False)


# In[151]:


c = c[['is_two_share','first_service_name']]


# In[152]:


c = c.set_index('first_service_name', inplace=False)


# In[170]:


## график
c.sort_values(by='is_two_share').plot(kind='bar',
    title='Доля пользователей с 2 и более заказами по сервису ',
    rot = 50,
    xlabel = 'Сервис первого заказа',
    ylabel = 'Доля пользователей с 2 и более заказами',
)
plt.show()


# In[173]:


c['is_two_share'].mean()


# Чаще всего в Яндекс.Афишу возвращаются пользователи - посетившие первым мероприятием выставки, сделавшие первый заказ с компьютера, сделавшие заказ через "Зе Бест!"
# Успешная точка входа через сервис "Зе Бест!" - в этом сегменте клиенты совершают повторный заказ чаще, чем в среднем по выборке.

# ---
# 
# **Задача 4.1.3.** Опираясь на выводы из задач выше, проверьте продуктовые гипотезы:
# 
# - **Гипотеза 1.** Тип мероприятия влияет на вероятность возврата на Яндекс Афишу: пользователи, которые совершили первый заказ на спортивные мероприятия, совершают повторный заказ чаще, чем пользователи, оформившие свой первый заказ на концерты.
# - **Гипотеза 2.** В регионах, где больше всего пользователей посещают мероприятия, выше доля повторных заказов, чем в менее активных регионах.
# 
# ---

# Гипотеза 1 - не подтверждается, пользователи, которые совершили первый заказ на концерты чаще совершают повторный заказ, чем пользователи, оформившие свой первый заказ на спортивные мероприятия
# 

# Гипотеза 2 - подтверждается. В регионах, где больше всего пользователей посещают мероприятия доля повторных заказов (средняя по топ 15) - 60.5%, что выше, чем доля повторных заказов в регионах, где меньше всего посещают мероприятия (50.5%)
# 

# ---
# 
# #### 4.2. Исследование поведения пользователей через показатели выручки и состава заказа
# 
# Изучите количественные характеристики заказов пользователей, чтобы узнать среднюю выручку сервиса с заказа и количество билетов, которое пользователи обычно покупают.
# 
# Эти метрики важны не только для оценки выручки, но и для оценки вовлечённости пользователей. Возможно, пользователи с более крупными и дорогими заказами более заинтересованы в сервисе и поэтому чаще возвращаются.
# 
# ---
# 
# **Задача 4.2.1.** Проследите связь между средней выручкой сервиса с заказа и повторными заказами.
# 
# - Постройте сравнительные гистограммы распределения средней выручки с билета (`avg_revenue_rub`):
#     - для пользователей, совершивших один заказ;
#     - для вернувшихся пользователей, совершивших 2 и более заказа.
# - Ответьте на вопросы:
#     - В каких диапазонах средней выручки концентрируются пользователи из каждой группы?
#     - Есть ли различия между группами?
# 
# Текст на сером фоне:
#     
# **Рекомендация:**
# 
# 1. Используйте одинаковые интервалы (`bins`) и прозрачность (`alpha`), чтобы визуально сопоставить распределения.
# 2. Задайте параметру `density` значение `True`, чтобы сравнивать форму распределений, даже если число пользователей в группах отличается.
# 
# ---
# 

# In[62]:


client_profile_f_1 = client_profile_f.loc[client_profile_f['is_two'] == True]
plt.figure(figsize=(20, 5))
sns.histplot(data=client_profile_f_1, x='avg_revenue_rub', bins = 70).set(title = 'Распределение значений средней выручки с 2+ заказами', 
                                                                 ylabel = 'Частота',
                                                                 xlabel = 'Среднняя выручка'
                                                                 )
plt.show()


# In[63]:


client_profile_f_2 = client_profile_f.loc[client_profile_f['is_two'] == False]
plt.figure(figsize=(20, 5))
sns.histplot(data=client_profile_f_2, x='avg_revenue_rub', bins = 70).set(title = 'Распределение значений средней выручки с 1 заказом', 
                                                                 ylabel = 'Частота',
                                                                 xlabel = 'Среднняя выручка'
                                                                 )
plt.show()


# У клиентов совершивших один заказ пользователи концентрируются в диапазоне выручки от 0 до 1000 (в основном).
# У клиентов с 2 и более заказами аналогично.
# Различия есть в частоте пользователей - в сегменте с клиентами, у которых было 2+ заказов - клиентов больше

# ---
# 
# **Задача 4.2.2.** Сравните распределение по средней выручке с заказа в двух группах пользователей:
# 
# - совершившие 2–4 заказа;
# - совершившие 5 и более заказов.
# 
# Ответьте на вопрос: есть ли различия по значению средней выручки с заказа между пользователями этих двух групп?
# 
# ---
# 

# In[64]:


client_profile_f_3 = client_profile_f.loc[(client_profile_f['is_two'] == True) & (client_profile_f['is_five'] == False)]
plt.figure(figsize=(20, 5))
sns.histplot(data=client_profile_f_3, x='avg_revenue_rub', bins = 70).set(title = 'Распределение значений средней выручки с 2-5 заказами', 
                                                                 ylabel = 'Частота',
                                                                 xlabel = 'Среднняя выручка'
                                                                 )
plt.show()


# In[65]:


client_profile_f_4 = client_profile_f.loc[client_profile_f['is_five'] == True]
plt.figure(figsize=(20, 5))
sns.histplot(data=client_profile_f_4, x='avg_revenue_rub', bins = 70).set(title = 'Распределение значений средней выручки с 5+ заказами', 
                                                                 ylabel = 'Частота',
                                                                 xlabel = 'Среднняя выручка'
                                                                 )
plt.show()


# Разница между текущими 2мя сегментами есть: распределение средней выручки по клиентам с 5+ заказами смещено ближе к значению 500, в то время как в сегменте 2-4 заказа - смещение ближе к началу отрезка - до 500. Что говорит нам о том, что в первом сегменте средняя выручка больше, чем во втором.

# ---
# 
# **Задача 4.2.3.** Проанализируйте влияние среднего количества билетов в заказе на вероятность повторной покупки.
# 
# - Изучите распределение пользователей по среднему количеству билетов в заказе (`avg_tickets_count`) и опишите основные наблюдения.
# - Разделите пользователей на несколько сегментов по среднему количеству билетов в заказе:
#     - от 1 до 2 билетов;
#     - от 2 до 3 билетов;
#     - от 3 до 5 билетов;
#     - от 5 и более билетов.
# - Для каждого сегмента подсчитайте общее число пользователей и долю пользователей, совершивших повторные заказы.
# - Ответьте на вопросы:
#     - Как распределены пользователи по сегментам — равномерно или сконцентрировано?
#     - Есть ли сегменты с аномально высокой или низкой долей повторных покупок?
# 
# ---

# In[66]:


#категоризация по среднему кол-ву билетов в заказе
client_profile_f['cnt_tickets_cat'] = pd.cut(client_profile_f['avg_cnt_per_order'], bins=[0, 2, 3, 5,12],                                              labels=["1-2", "2-3", "3-5", "5+"], right=False)


# In[67]:


#для проверки корректности категоризации
#a = client_profile_f.loc[client_profile_f['cnt_tickets_cat'] == '1-2']
#print(a['avg_cnt_per_order'].sort_values().unique())


# In[68]:


total_count = client_profile_f['user_id'].count()
cat_count = client_profile_f.groupby('cnt_tickets_cat')['user_id'].count()
cat_count =  pd.DataFrame(cat_count) 
cast_share = round(cat_count/total_count,2)*100
cast_share.reset_index()
cat_count.reset_index()


# In[69]:


cat_stat = pd.merge(cast_share, cat_count, left_on = ['cnt_tickets_cat'], right_on =  ['cnt_tickets_cat'], how = 'left')
cat_stat.reset_index()


# In[70]:


cat_stat = cat_stat.rename(columns={'user_id_x': 'share', 'user_id_y': 'cnt'})


# In[71]:


cat_stat


# Пользователи распределены неравномерно - в сегментах с 2-3 билетами и 3-5 билетами находится 86% клиентов.
# Самая низкая доля повторных покупок в сегменте 5+

# ---
# 
# #### 4.3. Исследование временных характеристик первого заказа и их влияния на повторные покупки
# 
# Изучите временные параметры, связанные с первым заказом пользователей:
# 
# - день недели первой покупки;
# - время с момента первой покупки — лайфтайм;
# - средний интервал между покупками пользователей с повторными заказами.
# 
# ---
# 
# **Задача 4.3.1.** Проанализируйте, как день недели, в которой была совершена первая покупка, влияет на поведение пользователей.
# 
# - По данным даты первого заказа выделите день недели.
# - Для каждого дня недели подсчитайте общее число пользователей и долю пользователей, совершивших повторные заказы. Результаты визуализируйте.
# - Ответьте на вопрос: влияет ли день недели, в которую совершена первая покупка, на вероятность возврата клиента?
# 
# ---
# 

# In[72]:


client_profile_f['day_of_week']= client_profile_f['min_order_ts'].dt.weekday


# In[73]:


cnt_day_of_week = client_profile_f.groupby('day_of_week')['user_id'].count()
cnt_day_of_week


# In[74]:


second_orders = client_profile_f.loc[(client_profile_f['is_two']==True)]
for_days_share = second_orders.groupby('day_of_week')['user_id'].count()
for_days_share


# In[75]:


second_orders_days_share = round(for_days_share/cnt_day_of_week*100,2)


# In[76]:


second_orders_days_share


# In[77]:


day_of_week_data = pd.merge(cnt_day_of_week,second_orders_days_share,left_on = ['day_of_week'],                            right_on = ['day_of_week'], how = 'left').reset_index()
day_of_week_data = day_of_week_data.rename(columns={'user_id_x': 'cnt', 'user_id_y': 'share'})


# In[78]:


day_of_week_data


# In[79]:


day_of_week_data.plot(
  kind='bar',
  x='day_of_week',
  y='cnt',
  alpha=0.5,
  color='lightgreen',
  edgecolor='k',
  title='Число пользователей по дням недели',
  xlabel = 'День недели',
  ylabel='Кол-во пользователей',
  legend = False
)

plt.show()


# In[80]:


day_of_week_data.plot(
  kind='bar',
  x='day_of_week',
  y='share',
  alpha=0.5,
  color='lightgreen',
  edgecolor='k',
  title='Доля пользователей с повторными заказами по дням недели',
  xlabel = 'День недели',
  ylabel='Доля пользователей с повторными заказами',
  legend = False
)

plt.show()


# Выглядит так, будто сильной взаимосвязи между днем недели и возвращаемостью клиента нет.

# ---
# 
# **Задача 4.3.2.** Изучите, как средний интервал между заказами влияет на удержание клиентов.
# 
# - Рассчитайте среднее время между заказами для двух групп пользователей:
#     - совершившие 2–4 заказа;
#     - совершившие 5 и более заказов.
# - Исследуйте, как средний интервал между заказами влияет на вероятность повторного заказа, и сделайте выводы.
# 
# ---
# 

# In[81]:


second_orders.groupby('is_five')['date_difference_d'].mean()


# У пользователей с 5+ заказами среднее время между заказами в 2 раза меньше, чем у пользователей, совершивших 2-4 заказа.

# ---
# 
# #### 4.4. Корреляционный анализ количества покупок и признаков пользователя
# 
# Изучите, какие характеристики первого заказа и профиля пользователя могут быть связаны с числом покупок. Для этого используйте универсальный коэффициент корреляции `phi_k`, который позволяет анализировать как числовые, так и категориальные признаки.
# 
# ---
# 
# **Задача 4.4.1:** Проведите корреляционный анализ:
# - Рассчитайте коэффициент корреляции `phi_k` между признаками профиля пользователя и числом заказов (`total_orders`). При необходимости используйте параметр `interval_cols` для определения интервальных данных.
# - Проанализируйте полученные результаты. Если полученные значения будут близки к нулю, проверьте разброс данных в `total_orders`. Такое возможно, когда в данных преобладает одно значение: в таком случае корреляционный анализ может показать отсутствие связей. Чтобы этого избежать, выделите сегменты пользователей по полю `total_orders`, а затем повторите корреляционный анализ. Выделите такие сегменты:
#     - 1 заказ;
#     - от 2 до 4 заказов;
#     - от 5 и выше.
# - Визуализируйте результат корреляции с помощью тепловой карты.
# - Ответьте на вопрос: какие признаки наиболее связаны с количеством заказов?
# 
# ---

# In[82]:


client_profile_for_corr = client_profile_f[['first_order_device','first_order_region','first_service_name',                                                   'first_event_type','total_orders_cnt','is_two','is_five','cnt_tickets_cat','day_of_week']] 
correlation_matrix=client_profile_for_corr.phik_matrix(interval_cols = ['total_orders_cnt'],                                                         
                                                                bins = {'total_orders_cnt':3})

#а df.phik_matrix(interval_cols=['cost'], bins={'cost':5}) у


# In[83]:


print("Корреляция переменных с 'total_orders_cnt':")
print(correlation_matrix['total_orders_cnt'])


# In[84]:


sns.heatmap(correlation_matrix).set(title='Матрица корреляций')
plt.show()


# In[85]:


#категоризация по среднему кол-ву билетов в заказе
client_profile_f['cnt_tickets_cat_v2'] = pd.cut(client_profile_f['total_orders_cnt'], bins=[0, 2, 5, 153],                                              labels=["1", "2-4", "5+"], right=False)


# In[86]:


#для проверки корректности категоризации
a = client_profile_f.loc[client_profile_f['cnt_tickets_cat_v2'] == '5+']
print(a['total_orders_cnt'].sort_values().unique())


# In[87]:


client_profile_f.groupby('cnt_tickets_cat_v2')['user_id'].count()


# In[88]:


client_profile_f


# In[89]:


client_profile_for_corr = client_profile_f[['first_order_device','first_order_region','first_service_name',                                                   'first_event_type','cnt_tickets_cat_v2','is_two','is_five','day_of_week',                                           'avg_cnt_per_order','date_difference_d']] 
correlation_matrix=client_profile_for_corr.phik_matrix()

#а df.phik_matrix(interval_cols=['cost'], bins={'cost':5}) у


# In[90]:


correlation_matrix


# In[91]:


sns.heatmap(correlation_matrix).set(title='Матрица корреляций')
plt.show()


# Наиболее связанные параметры с кол-вом заказов - среднее кол-во билетов в заказе и средний промежуток между заказами.

# ### 5. Общий вывод и рекомендации
# 
# В конце проекта напишите общий вывод и рекомендации: расскажите заказчику, на что нужно обратить внимание. В выводах кратко укажите:
# 
# - **Информацию о данных**, с которыми вы работали, и то, как они были подготовлены: например, расскажите о фильтрации данных, переводе тенге в рубли, фильтрации выбросов.
# - **Основные результаты анализа.** Например, укажите:
#     - Сколько пользователей в выборке? Как распределены пользователи по числу заказов? Какие ещё статистические показатели вы подсчитали важным во время изучения данных?
#     - Какие признаки первого заказа связаны с возвратом пользователей?
#     - Как связаны средняя выручка и количество билетов в заказе с вероятностью повторных покупок?
#     - Какие временные характеристики влияют на удержание (день недели, интервалы между покупками)?
#     - Какие характеристики первого заказа и профиля пользователя могут быть связаны с числом покупок согласно результатам корреляционного анализа?
# - Дополните выводы информацией, которая покажется вам важной и интересной. Следите за общим объёмом выводов — они должны быть компактными и ёмкими.
# 
# В конце предложите заказчику рекомендации о том, как именно действовать в его ситуации. Например, укажите, на какие сегменты пользователей стоит обратить внимание в первую очередь, а какие нуждаются в дополнительных маркетинговых усилиях.

# **Информация о данных:** Данные были импортированы и проверены на дубликаты, были отфильтрованы и исключены из анализа 0.97% данных, являющихся выбросами (99 процентиль). Так как в исходных данных была информация не только о рублевых операциях, но и об опрерациях с казахской валютой, было необходимо добавить курс этой валюты к общему датафрейму, для того, чтобы вычислять дальнейшие значения выручки в рублевом эквиваленте.

# **Основные результаты анализа**:
# - в отфильтрованном датафрейме содержатся данные по 21 854 пользователях и их заказах.
# - для большинства клиентов концерты являются мероприятием входа на платформу - 44%. 
# - у бОльшего кол-ва пользователь регион проведения мероприятия по первому заказу - Каменевский регион, 32.7%.
# - большая часть клиентов начала пользоваться сервисом с мобильного устройства - 82.8%.
# - "Билеты без проблем" - наиболее частый сервис для совершения первого заказа.
# - у клиентов совершивших один заказ пользователи концентрируются в диапазоне выручки от 0 до 1000 (в основном). У клиентов с 2 и более заказами аналогично. Различия есть в частоте пользователей - в сегменте с клиентами, у которых было 2+ заказов - клиентов больше.
# - Пользователи распределены неравномерно - в сегментах с 2-3 билетами и 3-5 билетами находится 86% клиентов. Самая низкая доля повторных покупок в сегменте 5+.
# - день недели не влияет на удержание клиентов.
# - у пользователей с 5+ заказами среднее время между заказами в 2 раза меньше, чем у пользователей, совершивших 2-4 заказа.
# - согласно результатам корреляционного анализа, наиболее связанные параметры с кол-вом заказов - среднее кол-во билетов в заказе и средний промежуток между заказами.

# Исходя из проведенного анализа, могу порекоммендовать команде маректинга обратить внимание на клиентов из сегмента 5+, по которым доля повторных покупок аномально низкая, возможно необходимо предложить таким клиентам спциальные условия при оформлении крупного заказа.

# ### 6. Финализация проекта и публикация в Git
# 
# Когда вы закончите анализировать данные, оформите проект, а затем опубликуйте его.
# 
# Выполните следующие действия:
# 
# 1. Создайте файл `.gitignore`. Добавьте в него все временные и чувствительные файлы, которые не должны попасть в репозиторий.
# 2. Сформируйте файл `requirements.txt`. Зафиксируйте все библиотеки, которые вы использовали в проекте.
# 3. Вынести все чувствительные данные (параметры подключения к базе) в `.env`файл.
# 4. Проверьте, что проект запускается и воспроизводим.
# 5. Загрузите проект в публичный репозиторий — например, на GitHub. Убедитесь, что все нужные файлы находятся в репозитории, исключая те, что в `.gitignore`. Ссылка на репозиторий понадобится для отправки проекта на проверку. Вставьте её в шаблон проекта в тетрадке Jupyter Notebook перед отправкой проекта на ревью.

# **Вставьте ссылку на проект в этой ячейке тетрадки перед отправкой проекта на ревью.**
