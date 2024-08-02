import unittest
from unittest.mock import patch, MagicMock
from marketinggpt.tools.policy_search import policy_search_tool

class TestPolicySearch(unittest.TestCase):
    
    @patch('marketinggpt.tools.policy_search.PineconeVectorStore')
    def test_policy_search_valid_input(self, MockPineconeVectorStore):
        # Setup mock
        mock_vectorstore = MockPineconeVectorStore.return_value
        mock_retriever = mock_vectorstore.as_retriever.return_value
        mock_retriever.invoke.return_value = [{"title": "Policy 1"}, {"title": "Policy 2"}]
        
        # Test input
        query = "hihi"
        expected_output = [{"title": "Policy 1"}, {"title": "Policy 2"}]
        
        # Call the function
        result = policy_search_tool(query)
        
        # Assertions
        self.assertEqual(result, expected_output)
        
         
    @patch('marketinggpt.tools.policy_search.PineconeVectorStore')
    def test_policy_search_no_results(self, MockPineconeVectorStore):
        # Setup mock
        mock_vectorstore = MockPineconeVectorStore.return_value
        mock_retriever = mock_vectorstore.as_retriever.return_value
        mock_retriever.invoke.return_value = []
        
        # Test input
        query = "nonexistent policy"
        expected_output = []
        
        # Call the function
        result = policy_search_tool(query)
        
        # Assertions
        self.assertEqual(result, expected_output)


    @patch('marketinggpt.tools.policy_search.PineconeVectorStore')
    def test_policy_search_error_handling(self, MockPineconeVectorStore):
        # Setup mock
        mock_vectorstore = MockPineconeVectorStore.return_value
        mock_retriever = mock_vectorstore.as_retriever.return_value
        mock_retriever.invoke.side_effect = Exception("Some error occurred")
        
        # Test input
        query = "policy search query"
        
        # Call the function
        result = policy_search_tool(query)
        
        # Assertions
        self.assertIn("Some error occurred", result)

if __name__ == '__main__':
    unittest.main()
