#include <fstream>
#include <cstring>

void WriteLargeFile(const std::string &filename)
{
    const long long size = 2LL * 1024 * 1024 * 1024;
    const int bufferSize = 1024 * 1024;
    char buffer[bufferSize];
    std::memset(buffer, 0, bufferSize);

    std::ofstream outputFile(filename, std::ios::binary);
    for (long long i = 0; i < size; i += bufferSize)
    {
        outputFile.write(buffer, bufferSize);
    }
    outputFile.close();
}

int main()
{
    WriteLargeFile("largefile.txt");
    return 0;
}
