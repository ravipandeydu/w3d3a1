#!/usr/bin/env python3
"""
Test Runner for LLM Inference Calculator

Runs all tests and provides a summary of results.
"""

import unittest
import sys
import os
from io import StringIO


def run_tests(verbosity=2):
    """Run all tests and return results."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Capture output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity)
    result = runner.run(suite)
    
    # Print results
    output = stream.getvalue()
    print(output)
    
    return result


def main():
    """Main test runner."""
    print("🧪 LLM Inference Calculator - Test Suite")
    print("=" * 50)
    
    # Check if we can import the modules
    try:
        from src import LLMInferenceCalculator
        print("✅ Successfully imported calculator modules")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    
    # Run tests
    print("\n🚀 Running tests...")
    result = run_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('\n')[-2]}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('\n')[-2]}")
    
    # Overall result
    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💔 Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())