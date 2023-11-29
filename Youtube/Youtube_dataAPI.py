import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import os

def youtube_dataAPI():

    # YouTube API 초기화
    youtube_api_key = os.getenv("YOUTUBE_API_KEY").split(",")[7]
    print(youtube_api_key)

    youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    def date_ranges(start_year, end_year, end_month=12, end_day=31, step=30):
        """날짜 범위 생성 함수"""
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, end_month, end_day)
        while start_date < end_date:
            yield start_date, start_date + timedelta(days=step)
            start_date += timedelta(days=step)

    def get_video_stats(video_ids):
        """주어진 비디오 ID 목록에 대한 통계 정보를 가져옵니다."""
        stats_request = youtube.videos().list(
            part="statistics",
            id=','.join(video_ids)
        )
        stats_response = stats_request.execute()
        stats = {item['id']: item['statistics'] for item in stats_response.get('items', [])}
        return stats

    def get_channel_stats(channel_ids):
        """주어진 채널 ID 목록에 대한 통계 정보를 가져옵니다."""
        try:
            channel_stats_request = youtube.channels().list(
                part="statistics",
                id=','.join(channel_ids)
            )
            channel_stats_response = channel_stats_request.execute()
            channel_stats = {item['id']: item['statistics'] for item in channel_stats_response.get('items', [])}
            return channel_stats
        except HttpError as e:
            print(f"An HTTP error occurred: {e.resp.status} {e.content}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
    search_keyword = "stock OR Finance OR analysis OR investment OR NASDAQ OR S&P OR US market"

    # Calculate API Quota Cost
    SEARCH = 100
    VIDEO = 1
    CHANNEL = 1

    video_data = []
    channel_ids = set()  # 중복을 방지하기 위해 채널 ID를 저장하는 집합
    start_year = 2017 # API 제한 때문에 2년씩 해야 합니다.
    end_year = 2019 # API 제한 때문에 2년씩 해야 합니다.

    # 시계열 반복문을 사용하여 2017년부터 2024년까지의 데이터를 수집합니다.
    search_count, video_count, channel_count = 0, 0, 0
    for start, end in date_ranges(start_year, end_year):
        search_request = youtube.search().list(
            part="snippet",
            q=search_keyword,
            maxResults=50,
            type="video",
            publishedAfter=start.isoformat() + "Z",
            publishedBefore=end.isoformat() + "Z",
            videoDuration="medium"
        )
        search_response = search_request.execute()
        search_count += SEARCH

        video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
        video_stats = get_video_stats(video_ids)
        video_count += len(video_ids) * VIDEO
        
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']
            stats = video_stats.get(video_id, {})
            
            # 추후 채널 통계 정보를 가져오기 위해 채널 ID를 저장합니다.
            channel_ids.add(item['snippet']['channelId'])

            video_data.append({
                'video_id': video_id,
                'Title': item['snippet']['title'],
                'Video Link': f"https://www.youtube.com/watch?v={video_id}",
                'Publish Date': item['snippet']['publishedAt'],
                'View Count': stats.get('viewCount'),
                'Like Count': stats.get('likeCount'),
                'Channel ID': item['snippet']['channelId'],
                'Subscriber Count' : None,
                'Video Count' : None,
                'Channel Views' : None
            })

    video_data = pd.DataFrame(video_data)

    channel_data = []
    channel_list = list(channel_ids)
    for channel_id in channel_list:
        channel_stats_request = youtube.channels().list(
            part="statistics",
            id=channel_id)
        
        channel_count += CHANNEL
        channel_stats_response = channel_stats_request.execute()
        channel_data.append({item['id']: item['statistics'] for item in channel_stats_response.get('items', [])})

    # channel_stats 를 DataFrame으로 변환
    combined_data = {}
    for entry in channel_data:
        for channel_id, stats in entry.items():
            # hiddenSubscriberCount 키를 제거
            stats.pop('hiddenSubscriberCount', None)
            combined_data[channel_id] = stats

    # DataFrame으로 변환
    channel_stats_df = pd.DataFrame.from_dict(combined_data, orient='index').reset_index()
    channel_stats_df.columns = ['Channel ID', 'Channel View', 'Subscriber Count', 'Video Count']

    # video_data와 channel_stats_df를 'Channel ID'를 기준으로 조인
    merged_data = pd.merge(video_data, channel_stats_df, on='Channel ID', how='inner')
    merged_data['Subscriber Count'] = merged_data['Subscriber Count_y']
    merged_data['Video Count'] = merged_data['Video Count_y']
    merged_data['Channel Views'] = merged_data['Channel Views']

    # 더 이상 필요 없는 컬럼 제거
    merged_data.drop(['Subscriber Count_x', 'Subscriber Count_y', 'Video Count_x', 'Video Count_y', 'Channel Views'], axis=1, inplace=True)

    # 결과 확인
    print("cost : ", search_count, video_count, channel_count, search_count + video_count + channel_count)
    merged_data.to_csv(f"youtube_{start_year}_{end_year}_merged.csv", index=False)
    
    return merged_data

def youtube_filter(df: pd.DataFrame):
    df.drop_duplicates(subset=['video_id'], inplace=True)
    df = df[df['View Count'] > 1000]
    df = df[df['Like Count'] != 0]
    df = df[df['Subscriber Count'] > 100]
    df = df[df['Video Count'] > 10]
    df = df[df['Channel View'] > 10000]
    df.sort_values('Publish Date', inplace=True)
    
    return df
    