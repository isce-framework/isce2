#include <fstream>
#include <sstream>
#include <RelaxIV.h>
#include <relaxIVdriver.h>


using namespace MCFClass_di_unipi_it;

template<class T>
inline T ABS( const T x )
{
 return( x >= T( 0 ) ? x : -x );
}

using namespace std;
//extern void SetParam( MCFClass *mcf );

template<class T>
static inline void str2val( const char* const str , T &sthg )
{
 istringstream( str ) >> sthg;
 }

/*--------------------------------------------------------------------------*/
// This function skips comment line in a input stream, where comment line is 
// // marked by an initial '#' character
//
void SkipComments( ifstream &iParam , string &buf )
 {
  do {
    iParam >> ws;
      getline( iParam , buf );
       }
        while( buf[ 0 ] == '#' );
        }




void SetParam( MCFClass *mcf )
{
 ifstream iParam( "config.txt" ); 
 if( ! iParam.is_open() )
  return;

 string buf;
 int num;
 SkipComments( iParam , buf );
 str2val( buf.c_str(), num );        // get number of int parameters

 for( int i = 0 ; i < num ; i++ ) {  // read all int parameters
  int param , val;
  
  SkipComments( iParam , buf );
  str2val( buf.c_str(), param );     // parameter name
  
  SkipComments( iParam , buf );
  str2val( buf.c_str(), val );       // parameter value

  mcf->SetPar( param , val );

  }  // end( for( i ) )

 SkipComments( iParam , buf );
 str2val( buf.c_str() , num );       // get number of double parameters

 for( int i = 0 ; i < num ; i++ ) {  // read all double parameters
  int param;
  double val;
  SkipComments( iParam , buf );
  str2val( buf.c_str(), param );     // parameter name
  
  SkipComments( iParam , buf );
  str2val( buf.c_str() , val );      // parameter value
  
  mcf->SetPar( param , val );

  }  // end( for( i ) )
 }  // end( SetParam )


vector<int> driver(char *fileName)
{
 ifstream iFile(fileName);
 if(  !iFile ) {
  cerr << "ERROR: opening input file " << fileName << endl;
  return std::vector<int>();
  }

  // construct the solver - - - - - - - - - - - - - - - - - - - - - - - - - -

  MCFClass *mcf = new RelaxIV();

  mcf->SetMCFTime();  // do timing

  // load the network - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  cout << "Loading Network: " << fileName  << endl;
  mcf->LoadDMX( iFile );

  // set "reasonable" values for the epsilons, if any - - - - - - - - - - - -

  cout << "Running Relax IV" << endl;
  MCFClass::FNumber eF = 1;
  for( register MCFClass::Index i = mcf->MCFm() ; i-- ; )
   eF = max( eF , ABS( mcf->MCFUCap( i ) ) );

  for( register MCFClass::Index i = mcf->MCFn() ; i-- ; )
   eF = max( eF , ABS( mcf->MCFDfct( i ) ) );   

  MCFClass::CNumber eC = 1;
  for( register MCFClass::Index i = mcf->MCFm() ; i-- ; )
   eC = max( eC , ABS( mcf->MCFCost( i ) ) );

  mcf->SetPar( RelaxIV::kEpsFlw, (double) numeric_limits<MCFClass::FNumber>::epsilon() * eF *
		  mcf->MCFm() * 10);  // the epsilon for flows

  mcf->SetPar( RelaxIV::kEpsCst, (double) numeric_limits<MCFClass::CNumber>::epsilon() * eC *
		  mcf->MCFm() * 10);  // the epsilon for costs

  
  // set other parameters from configuration file (if any)- - - - - - - - - -

   SetParam( mcf );
  
  // solver call- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

   mcf->SolveMCF();

  // output results - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  std::vector<int> retVal;
  switch( mcf->MCFGetStatus() ) {
   
   case( MCFClass::kOK ):
    cout << "Optimal Objective Function value = " << mcf->MCFGetFO() << endl;

    double tu , ts;
    mcf->TimeMCF( tu , ts );
    cout << "Solution time (s): user " << tu << ", system " << ts << endl;
    {
     if( ( numeric_limits<MCFClass::CNumber>::is_integer == 0 ) ||
	 ( numeric_limits<MCFClass::FNumber>::is_integer == 0 ) ) {
      cout.setf( ios::scientific, ios::floatfield );
      cout.precision( 12 );
      }
      
     MCFClass::FRow x = new MCFClass::FNumber[ mcf->MCFm() ];
     mcf->MCFGetX( x );
     for( MCFClass::Index i = 0 ; i < mcf->MCFm() ; i++ )
        retVal.push_back(x[i]);

     // check solution
     mcf->CheckPSol();
     mcf->CheckDSol();
     delete( mcf );
     return (retVal);

    }
    break;
   case( MCFClass::kUnfeasible ):
    cout << "MCF problem unfeasible." << endl;
    break;
   case( MCFClass::kUnbounded ):
    cout << "MCF problem unbounded." << endl;
    break;
   default:
    cout << "Error in the MCF solver." << endl;
   }

  // output the problem in MPS format - - - - - - - - - - - - - - - - - - - -
  /*
  if( argc > 2 ) {
   ofstream oFile( argv[ 2 ] );
   mcf->WriteMCF( oFile , MCFClass::kMPS );
   } */

  // destroy the object - - - - - - - - - - - - - - - - - - - - - - - - - - -

  delete( mcf );
  return std::vector<int>();
  
  
 // terminate - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

}
