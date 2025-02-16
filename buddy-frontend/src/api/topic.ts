import axios from "axios";
import { Topic, ChapterContent } from "../store/topicAndContentSlice";

// Create an axios instance with default configuration.
const apiClient = axios.create({
  baseURL: "http://localhost:5000",
  headers: {
    "Content-Type": "application/json",
  },
});

// Function to fetch all topics
export const getTopics = async (): Promise<Topic[]> => {
  try {
    const response = await apiClient.get<Topic[]>("/topics");
    return response.data;
  } catch (error) {
    return handleAxiosError(error);
  }
};

// Function to create a new topic
export const createTopic = async (topicName: string): Promise<Topic> => {
  try {
    const response = await apiClient.post<Topic>("/topics", {
      topic_name: topicName,
    });
    return response.data;
  } catch (error) {
    return handleAxiosError(error);
  }
};

// Function to fetch topic content for a given topic
export const getTopicContent = async (
  topicName: string
): Promise<ChapterContent[]> => {
  try {
    const response = await apiClient.get<ChapterContent[]>(
      `/topics/${topicName}`
    );
    return response.data;
  } catch (error) {
    return handleAxiosError(error);
  }
};

// Function to create/update topic content for a specific topic.
// This function accepts topicName, chapter number, summary and tags.
export const createTopicContent = async (
  topicName: string,
  formData: FormData
): Promise<ChapterContent> => {
  try {
    const response = await apiClient.post<ChapterContent>(
      `/topics/${topicName}`,
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
      }
    );
    return response.data;
  } catch (error) {
    return handleAxiosError(error);
  }
};

// Utility function to handle axios errors
const handleAxiosError = (error: unknown): never => {
  if (axios.isAxiosError(error)) {
    console.error("Axios error:", error.message);
    if (error.response) {
      console.error("Response data:", error.response.data);
      console.error("Response status:", error.response.status);
    }
  } else {
    console.error("Unexpected error:", error);
  }
  throw error;
};
