import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export interface Topic {
  topic_name: string;
  uuid: string;
}

export interface ChapterContent {
  topic_name: string;
  chapter_num: number;
  summary: string;
  tags: string[];
}

interface TopicState {
  activeTopic?: Topic;
  topics: Topic[];
  topicContent: ChapterContent[];
}

const initialState: TopicState = {
  topics: [],
  topicContent: [],
};

const topicAndContentSlice = createSlice({
  name: "topicAndContent",
  initialState,
  reducers: {
    setTopics(state, action: PayloadAction<Topic[]>) {
      state.topics = action.payload;
    },
    setActiveTopic(state, action: PayloadAction<Topic>) {
      state.activeTopic = action.payload;
    },
    setTopicContent(state, action: PayloadAction<ChapterContent[]>) {
      state.topicContent = action.payload;
    },
    addTopic(state, action: PayloadAction<Topic>) {
      state.topics.push(action.payload);
    },
    addTopicContent(state, action: PayloadAction<ChapterContent>) {
      state.topicContent.push(action.payload);
    },
  },
});

export const {
  setTopics,
  setTopicContent,
  addTopic,
  addTopicContent,
  setActiveTopic,
} = topicAndContentSlice.actions;
export default topicAndContentSlice.reducer;
