#pragma once

#ifndef STREAM_UTILITY_H
#define STREAM_UTILITY_H

#include <cstring>
#include <iterator>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

enum SeekDirection {
	Seek_Set,
	Seek_Forward,
	Seek_Backward,
	Seek_End
};

class StreamUtility {
private:
	std::vector< uint8_t > & m_stream;
	std::vector< uint8_t > m_stream_internal;
	int32_t m_read_index;
	bool m_read_error;
	bool m_write_error;

public:
	StreamUtility();
	StreamUtility( std::vector< uint8_t > & stream );
	StreamUtility( const void * const buffer, int32_t size );
	StreamUtility( const StreamUtility & rhs );
	~StreamUtility();
	StreamUtility & operator =( const StreamUtility & rhs );
	void Clear();
	bool WasWriteError();
	bool WasReadError();
	void ClearReadError();
	void ClearWriteError();
	const std::vector<uint8_t>& GetStreamVector() const;
	const uint8_t * GetStreamPtr();
	int32_t GetStreamSize();
	int32_t GetWriteIndex();
	const uint8_t * GetReadStreamPtr();
	int32_t GetReadStreamSize();
	int32_t GetReadIndex();
	bool SeekRead( int32_t index, SeekDirection dir );
	int32_t Delete( int32_t index, int32_t count );
	std::string Read_Ascii( int32_t count );
	void Read(std::string &out);
	std::wstring Read_AsciiToUnicode( int32_t count );
	std::wstring Read_Unicode( int32_t count );
	std::string Read_UnicodeToAscii( int32_t count );
	StreamUtility Extract( int32_t index, int32_t count );

	template <typename type>
	type Read( bool peek = false ) {
		type val{0}; // TODO: Why did I add this zero-initialization?
		Read< type >( &val, 1, peek );
		return val;
	}

	template <typename type>
	void Read( type * out, int32_t count, bool peek = false )
	{
		if( count == 0 )
		{
			return;
		}
		if( m_read_error || m_read_index + ( count * sizeof( type ) ) > static_cast< int32_t >( m_stream.size() ) )
		{
			m_read_error = true;
		}
		else
		{
			memcpy( out, &m_stream[m_read_index], ( count * sizeof( type ) ) );
			if( !peek )
			{
				m_read_index += ( count * sizeof( type ) );
			}
		}
	}

  template <typename T>
  void Read(T &out, bool peek=false) {
    // TODO: Replace most uses of the value-returning Read with this one
    Read<T>(&out, 1, peek);
  }

	template <typename type>
	void Write( const std::vector< type > & val ) {
		Write< type >( val.empty() ? 0 : &val[0], static_cast< int32_t >( val.size() ) );
	}

  void Write(std::string_view str);

  template <typename T, typename = std::enable_if_t<!std::is_convertible_v<T, std::string_view>>>
  void Write(T val) {
    Write<T>( &val, 1 );
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_enum_v<T>>>
  void Write(const T * const input, int32_t count) {
    if (count) {
      std::copy( (uint8_t *)input, (count * sizeof(T)) + (uint8_t *)input, std::back_inserter( m_stream ) );
    }
  }

	template <typename type>
	void Insert( int32_t index, const std::vector< type > & val )
	{
		Insert< type >( index, val.empty() ? 0 : &val[0], static_cast< int32_t >( val.size() ) );
	}

	template <typename type>
	void Insert( int32_t index, type val )
	{
		Insert< type >( index, &val, 1 );
	}

	template <typename type>
	void Insert( int32_t index, const type * const input, int32_t count )
	{
		if( index >= static_cast< int32_t >( m_stream.size() ) )
		{
			m_stream.resize( static_cast< int32_t >( m_stream.size() ) + ( sizeof( type ) * count ) );
		}
		else
		{
			m_stream.resize( m_stream.size() + ( sizeof( type ) * count ) );
			memmove( &m_stream[index + sizeof( type ) * count], &m_stream[index], static_cast< int32_t >( m_stream.size() ) - index - sizeof( type ) * count );
		}
		memcpy( &m_stream[index], input, ( count * sizeof( type ) ) );
	}

	template <typename type>
	void Overwrite( int32_t index, const std::vector< type > & val )
	{
		Overwrite< type >( index, val.empty() ? 0 : &val[0], static_cast< int32_t >( val.size() ) );
	}

	template <typename type>
	void Overwrite( int32_t index, type val )
	{
		Overwrite< type >( index, &val, 1 );
	}

	template <typename type>
	void Overwrite( int32_t index, const type * const input, int32_t count )
	{
		if( static_cast< int32_t >( m_stream.size() ) < index + ( count * sizeof( type ) ) )
		{
			m_stream.resize( index + ( count * sizeof( type ) ) );
		}
		memcpy( &m_stream[index], input, count * sizeof( type ) );
	}

	template <typename type>
	void Fill( int32_t index, type value, int32_t count )
	{
		if( static_cast< int32_t >( m_stream.size() ) < index + ( sizeof( type ) * count ) )
		{
			m_stream.resize( index + ( sizeof( type ) * count ) );
		}
		type * stream = reinterpret_cast< type * >( &m_stream[index] );
		for( int32_t x = 0; x < count; ++x )
		{
			stream[x] = value;
		}
	}
};

//-----------------------------------------------------------------------------

std::string DumpToString( StreamUtility & stream_utility );
std::string DumpToString( const std::vector< uint8_t > & buffer );
std::string DumpToString( const void * stream, int32_t size );

//-----------------------------------------------------------------------------

#endif
